"""MonoQPM: split each polysemantic feature of a trained QPM into its monosemantic concepts.

Reference implementation for the MonoQPM pipeline (paper Sections 5.1-5.3). It takes a frozen
QPM -- a backbone plus a binary class/feature assignment ``W`` of shape ``[n_classes, n_features]``
-- and turns every feature into one or more concepts, without retraining anything.

For each feature ``d``:

1.  **Spline input (eq. 15-16).** Every training image is passed through the frozen model once, in
    evaluation mode. For each call of each ``nn.ReLU`` of the backbone, in network order, the
    fraction of spatial positions with a positive output is recorded per channel. Concatenated
    this is the ReLU code ``r`` (length 3392 for resnet18, 7104 for resnet34).

2.  **Samples of the feature.** The classes ``Y^d = {y : W[y, d] != 0}``, and of those every
    training image the model classifies correctly. No outlier removal, normalisation or
    downsampling.

3.  **Depth selection (eq. 17).** On at most 5000 mean-centred rows, the participation ratio of
    each ReLU call's channel block is computed, and the last local maximum strictly before the
    global one is kept: ``D = argmax_{l < l_max} PR(R^(l))``. Blocks ``0..D`` are kept.

4.  **Dimensionality reduction (Section 5.1).** PCA with ``floor(PR)`` components, then a
    one-component LDA against the class labels, giving one scalar per image.

5.  **Clustering (eq. 13).** For ``k = 1 .. min(30, ceil(|Y^d| / 2))`` a 1-D Gaussian mixture is
    initialised at the Fisher-Jenks natural breaks and fitted; the ``k`` minimising BIC is the
    number of concepts.

6.  **Class assignment (eq. 14).** Modes are relabelled along the greedy nearest-neighbour path
    through their means. A class joins every concept whose mean responsibility over that class's
    images is at least ``tau = 1 / (2 n_c)``. Concepts no class joined are dropped.

At inference the concepts are used by eq. 8 and eq. 9, implemented in ``monoqpm/conformal.py``:
a test image's feature value goes to the concept whose classes hold its largest logit, every
other concept slot of that feature holds ``mu_d``, and a class logit is the sum over its features
of the maximum over the concepts it owns. ``mu_d`` is the minimum of feature ``d`` over the
evaluated set. Because the predicted class is the global logit maximum, its concept always wins
the routing of every feature it owns, so **MonoQPM provably cannot change the top-1 prediction**.

Implementation notes, all verified against the authors' research implementation:

* ``jenks_breaks`` reimplements the Fisher-Jenks dynamic programme (same recursion, same ``>=``
  tie rule, same break convention) so that ``jenkspy`` is not a dependency. It is bit-identical
  to jenkspy 0.4.1 on 132 test cases; see ``tests/test_monoqpm.py``.
* ``PCA_SVD_SOLVER = "full"``. scikit-learn's default ``svd_solver="auto"`` selects the
  randomized solver at these shapes and does not seed it, which makes the concept assignment
  irreproducible run to run (three seeds agreed on the class partition of only 18-22 of 50
  features on a CUB200 resnet34 QPM, though on the number of concepts for 43-45 of them). The
  exact solver removes that variance; set ``"auto"`` to reproduce the original behaviour.
* Eq. 12 of the paper defines the participation ratio on covariance eigenvalues; the reference
  implementation computes ``(sum sigma)^2 / sum sigma^2`` over singular values and floors rather
  than rounds it. This module follows the implementation, which keeps roughly five times more PCA
  components and measures better; ``ParticipationRatio`` below documents both.
* A class with no correctly classified training image is dropped from that feature's rows.
"""
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture

SHARE_THRESHOLD_SCALE = 0.5  # "K5K": a class joins a mode if its mean responsibility >= 0.5 / k
MAX_COMPONENTS = 30
SWEEP_MAX_SAMPLES = 5000
SWEEP_SEED = 42
GMM_SEED = 42
GMM_MAX_ITER = 100
PCA_SVD_SOLVER = "full"  # "auto" is randomized and unseeded at these shapes; see the module docstring


# ----------------------------------------------------------------------------------------------
# 1. ReLU activation-fraction codes
# ----------------------------------------------------------------------------------------------

@torch.no_grad()
def relu_activation_fractions(model, images, **forward_kwargs):
    """One forward pass. Returns ``(codes [B, D], ranges, output)``.

    ``codes[:, s:e]`` for ``(s, e)`` in ``ranges`` is one ReLU call: per channel the fraction of
    spatial positions with output ``> 0``. Calls are in execution order. ``output`` is whatever
    the model returns for ``forward_kwargs`` (logits, or a list with the extras asked for).
    """
    chunks = []

    def hook(_module, _inputs, output):
        flat = output.reshape(output.shape[0], output.shape[1], -1)
        chunks.append((flat > 0).float().mean(dim=2))

    handles = [m.register_forward_hook(hook)
               for _, m in model.named_modules() if isinstance(m, torch.nn.ReLU)]
    try:
        out = model(images, **forward_kwargs)
    finally:
        for h in handles:
            h.remove()
    ranges, start = [], 0
    for c in chunks:
        ranges.append((start, start + c.shape[1]))
        start += c.shape[1]
    return torch.cat(chunks, dim=1), ranges, out


@torch.no_grad()
def extract_relu_codes(model, dataset, device, batch_size=64, num_workers=8, with_codes=True):
    """One deterministic pass over a dataset in index order (give it the evaluation transform).

    Returns a dict with ``codes [N, D]`` (or None) and ``ranges``, plus ``logits [N, C]``,
    ``features [N, F]`` (the head's normalized input) and ``labels [N]``.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers)
    model.eval()
    codes, logits, features, labels, ranges = [], [], [], [], None
    for images, target in loader:
        images = images.to(device)
        if with_codes:
            c, r, out = relu_activation_fractions(model, images, with_final_features=True)
            if ranges is None:
                ranges = r
            assert r == ranges, "ReLU call layout changed between batches"
            codes.append(c.cpu())
        else:
            out = model(images, with_final_features=True)
        logit, feature = out
        logits.append(logit.detach().float().cpu())
        features.append(feature.detach().float().cpu())
        labels.append(torch.as_tensor(target))
    return {"codes": torch.cat(codes).numpy().astype(np.float32) if with_codes else None,
            "ranges": ranges,
            "logits": torch.cat(logits).numpy().astype(np.float32),
            "features": torch.cat(features).numpy().astype(np.float32),
            "labels": torch.cat(labels).numpy().astype(np.int64)}


# ----------------------------------------------------------------------------------------------
# 2.-4. Per-feature scalar projection
# ----------------------------------------------------------------------------------------------

def participation_ratio(x):
    """Participation ratio of a centred matrix; 1.0 for an all-zero matrix.

    Paper eq. 12 writes it on the covariance eigenvalues, ``(sum lambda)^2 / sum lambda^2``. The
    reference implementation computes it on the SINGULAR values instead, which is what this
    returns: ``(sum sigma)^2 / sum sigma^2 = (sum sqrt(lambda))^2 / sum lambda``. The two are
    different measures of effective dimensionality; the singular-value one is always the larger
    and is the one the published numbers use. See the module docstring.
    """
    s = np.linalg.svd(x, compute_uv=False)
    denominator = float((s ** 2).sum())
    return 1.0 if denominator == 0 else float(s.sum() ** 2 / denominator)


def select_layers(codes, ranges, max_samples=SWEEP_MAX_SAMPLES, seed=SWEEP_SEED):
    """Index of the last ReLU call to keep and ``floor`` of the effective rank at that call."""
    if len(codes) > max_samples:
        keep = np.sort(np.random.default_rng(seed).choice(len(codes), size=max_samples, replace=False))
        codes = codes[keep]
    x = codes.astype(np.float64)
    x -= x.mean(axis=0, keepdims=True)
    ranks = np.array([participation_ratio(x[:, s:e]) for s, e in ranges])
    masked = ranks.copy()
    masked[int(np.argmax(ranks)):] = -np.inf
    selected = int(np.argmax(masked))
    return selected, int(ranks[selected])


def project_feature(codes, labels, ranges):
    """Scalar per row: layer sweep -> PCA(floor(effective rank)) -> LDA(1) with the labels.

    Returns ``(z, selected_call, n_components, fitted)``; ``fitted`` carries the transform so
    unseen samples can be projected into the same 1-D space (``project_with``).
    """
    selected, n_components = select_layers(codes, ranges)
    end = ranges[selected][1]
    x = codes[:, :end]
    pca = PCA(n_components=1 if len(np.unique(labels)) == 1 else n_components,
              svd_solver=PCA_SVD_SOLVER)
    scores = pca.fit_transform(x)
    if len(np.unique(labels)) == 1:
        return scores[:, 0], selected, 1, {"end": end, "pca": pca, "lda": None}
    lda = LinearDiscriminantAnalysis(n_components=1)
    z = lda.fit_transform(scores, labels)[:, 0]
    return z, selected, n_components, {"end": end, "pca": pca, "lda": lda}


def project_with(fitted, codes):
    """Apply a fitted ``project_feature`` transform to new ReLU codes."""
    scores = fitted["pca"].transform(codes[:, :fitted["end"]])
    return scores[:, 0] if fitted["lda"] is None else fitted["lda"].transform(scores)[:, 0]


# ----------------------------------------------------------------------------------------------
# 5. Jenks-initialised 1-D Gaussian mixture, BIC-selected
# ----------------------------------------------------------------------------------------------

def jenks_breaks(values, n_classes):
    """Fisher-Jenks natural breaks, identical to ``jenkspy.jenks_breaks``.

    Returns ``n_classes + 1`` values: the minimum, the upper bound (largest member) of classes
    ``1 .. n_classes - 1``, and the maximum.
    """
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())
    n = len(v)
    if n_classes < 1 or n_classes > len(np.unique(v)):
        raise ValueError("n_classes must be between 1 and the number of unique values")
    lower = np.ones((n, n_classes), dtype=np.int64)  # 1-based lower class limits
    cost = np.full((n, n_classes), np.inf)
    v_sq = v * v
    for l in range(n):
        # Segments [lcl, l] for lcl = l, l-1, ..., 0, accumulated in that order as in _jenks.c.
        lcl = np.arange(l, -1, -1)
        w = np.arange(1, l + 2, dtype=np.float64)
        s = np.cumsum(v[l::-1])
        s2 = np.cumsum(v_sq[l::-1])
        variance = s2 - (s * s) / w
        cost[l, 0] = variance[-1]
        lower[l, 0] = 1
        if l == 0:
            continue
        i4 = lcl[:-1] - 1  # lcl >= 1 only
        for j in range(1, n_classes):
            candidates = variance[:-1] + cost[i4, j - 1]
            # The C loop updates on ">=", so among equal minima the last one (smallest lcl) wins.
            best = len(candidates) - 1 - int(np.argmin(candidates[::-1]))
            cost[l, j] = candidates[best]
            lower[l, j] = lcl[best] + 1
    breaks = [v[0]] * (n_classes + 1)
    breaks[n_classes] = v[-1]
    m, jj = n, n_classes
    while jj > 1:
        breaks[jj - 1] = v[lower[m - 1, jj - 1] - 2]
        m = lower[m - 1, jj - 1] - 1
        jj -= 1
    return breaks


def fit_jenks_gmm(z, n_components):
    """``GaussianMixture`` on 1-D data initialised at the Fisher-Jenks natural breaks (eq. 13)."""
    breaks = jenks_breaks(z, n_components)
    means = []
    for i in range(n_components):
        segment = z[(z >= breaks[i]) & (z <= breaks[i + 1])]
        means.append(segment.mean() if len(segment) else (breaks[i] + breaks[i + 1]) / 2)
    gmm = GaussianMixture(n_components=n_components, means_init=np.array(means)[:, None],
                          covariance_type="full", max_iter=GMM_MAX_ITER, random_state=GMM_SEED)
    return gmm.fit(z[:, None])


def mode_order(means):
    """``argsort`` of the greedy nearest-neighbour path through the mode means."""
    k = len(means)
    if k <= 1:
        return np.array([0])
    distance = np.abs(means[:, None] - means[None, :])
    path, visited = [0], {0}
    while len(path) < k:
        nearest, best = -1, np.inf
        for i, d in enumerate(distance[path[-1]]):
            if i not in visited and d < best:
                best, nearest = d, i
        path.append(nearest)
        visited.add(nearest)
    return np.argsort(path)


# ----------------------------------------------------------------------------------------------
# 6. Class-to-meaning assignment
# ----------------------------------------------------------------------------------------------

def cluster_classes(z, labels):
    """``({class: mode ids}, k, fitted)`` of the BIC-best mixture.

    ``fitted`` carries the mixture and the mode permutation, so the mode responsibilities of an
    unseen sample can be computed in the same numbering (``mode_probabilities``).
    """
    classes = np.unique(labels)
    k_max = min(MAX_COMPONENTS, int(np.ceil(len(classes) / 2)))
    gmms = [fit_jenks_gmm(z, k) for k in range(1, k_max + 1)]
    bics = [gmm.bic(z[:, None]) for gmm in gmms]
    gmm = gmms[int(np.argmin(bics))]
    k = gmm.n_components
    order = mode_order(gmm.means_[:, 0])
    probas = gmm.predict_proba(z[:, None])[:, order]
    threshold = SHARE_THRESHOLD_SCALE / k
    assignment = {int(c): np.flatnonzero(probas[labels == c].mean(axis=0) >= threshold)
                  for c in classes}
    return assignment, k, {"gmm": gmm, "order": order}


def mode_probabilities(fitted, z):
    """``[N, k]`` mixture responsibilities in the split's mode numbering."""
    return fitted["gmm"].predict_proba(np.asarray(z)[:, None])[:, fitted["order"]]


def invert_assignment(assignment):
    """``{class: modes}`` -> ``{tuple(classes): array(modes)}``, modes renumbered contiguously."""
    by_mode = defaultdict(list)
    for c, modes in assignment.items():
        for m in modes:
            by_mode[int(m)].append(c)
    remap = {m: i for i, m in enumerate(sorted(by_mode))}
    inverted = defaultdict(list)
    for m, cls in by_mode.items():
        inverted[tuple(cls)].append(remap[m])
    return {cls: np.array(modes) for cls, modes in inverted.items()}


# ----------------------------------------------------------------------------------------------
# The split object
# ----------------------------------------------------------------------------------------------

class MeaningSplit:
    """One slot per discovered concept.

    ``splitting[f]`` is ``{tuple(classes): array(meaning ids)}`` (``{}`` for a feature no class
    uses). ``slot_to_orig[s]`` is the original feature of slot ``s``; slots of one feature are
    contiguous and ordered by meaning id.
    """

    def __init__(self, splitting, info=None, models=None):
        self.splitting = list(splitting)
        self.info = info or {}
        # Per-feature fitted transform + mixture; only present when the split was computed with
        # keep_models=True, and only needed to route unseen samples by their own mode.
        self.models = models
        self.n_feats_init = len(self.splitting)
        self.n_meanings = np.array([max(1, len(self._meanings(f))) for f in range(self.n_feats_init)])
        self.first_slot = np.concatenate([[0], np.cumsum(self.n_meanings)[:-1]])
        self.slot_to_orig = np.repeat(np.arange(self.n_feats_init), self.n_meanings)
        self.total_features = int(self.n_meanings.sum())

    def _meanings(self, f):
        return set().union(*[set(int(m) for m in modes) for modes in self.splitting[f].values()])

    def split_weight(self, weight):
        """``W [C, F]`` -> ``W_mono [C, n_concepts]``: a class keeps its weight in every concept it owns."""
        w = np.asarray(weight.detach().cpu() if isinstance(weight, torch.Tensor) else weight)
        out = np.zeros((w.shape[0], self.total_features), dtype=w.dtype)
        for f in range(self.n_feats_init):
            for classes, modes in self.splitting[f].items():
                rows = list(classes)
                for m in modes:
                    slot = self.first_slot[f] + int(m)
                    assert not out[rows, slot].any(), (f, classes, m)
                    out[rows, slot] = w[rows, f]
        return out

    def route_masks(self, logits):
        """MaxLog routing: ``[N, total]`` bool, where a sample's value of the original feature lands."""
        logits = np.asarray(logits)
        masks = np.zeros((len(logits), self.total_features), dtype=bool)
        for f in range(self.n_feats_init):
            k = self.n_meanings[f]
            start = self.first_slot[f]
            if k == 1:
                masks[:, start] = True
                continue
            scores = np.zeros((len(logits), k))
            for classes, modes in self.splitting[f].items():
                best = logits[:, list(classes)].max(axis=1)
                for m in modes:
                    scores[:, int(m)] = best
            masks[:, start:start + k] = scores == scores.max(axis=1, keepdims=True)
        return masks

    def split_features(self, features, logits, inactive=None):
        """``[N, F]`` -> ``[N, total]``: routed values; every other slot holds ``inactive[f]``.

        ``inactive`` defaults to ``mu_d``, the per-feature minimum of ``features``; pass a ``[F]``
        vector to use other constants.
        """
        features = np.asarray(features)
        if inactive is None:
            inactive = features.min(axis=0)
        masks = self.route_masks(logits)
        gathered = features[:, self.slot_to_orig]
        out = np.broadcast_to(np.asarray(inactive)[self.slot_to_orig], gathered.shape).astype(features.dtype)
        out = np.where(masks, gathered, out)
        return out

    def summary(self):
        extra = self.n_meanings - 1
        return {"n_features_in": int(self.n_feats_init), "n_meaning_slots": self.total_features,
                "n_split_features": int((extra > 0).sum()),
                "extra_meanings_per_feature_mean": float(extra.mean()),
                "meanings_per_feature": self.n_meanings.tolist()}

    def save(self, path, with_models=True):
        """The fitted models are large (~100 MB); pass ``with_models=False`` for a lean file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"splitting": self.splitting, "info": self.info,
                         "models": self.models if with_models else None}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        return cls(state["splitting"], state.get("info"), state.get("models"))


# ----------------------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------------------

def compute_meaning_split(codes, ranges, logits, labels, class_feature_weight, verbose=True,
                          keep_models=False):
    """Run steps 2-6 for every feature. ``class_feature_weight`` is ``W [C, F]`` (nonzero = assigned).

    With ``keep_models`` the fitted per-feature transform and mixture are attached as
    ``split.models``, which is what lets an unseen sample be routed by its OWN mixture
    responsibility (``sample_mode_slots``) instead of by the class logits of paper eq. 8.
    """
    w = np.asarray(class_feature_weight.detach().cpu() if isinstance(class_feature_weight, torch.Tensor)
                   else class_feature_weight)
    labels = np.asarray(labels)
    logits = np.asarray(logits)
    correct = logits.max(axis=1) == logits[np.arange(len(labels)), labels]
    assert correct.mean() > 0.5, f"training accuracy {correct.mean():.3f} <= 0.5"
    splitting, models = [], []
    info = {"selected_layer": [], "pca_components": [], "n_modes": [], "n_rows": []}
    for f in range(w.shape[1]):
        classes = np.flatnonzero(w[:, f] != 0)
        rows = [np.flatnonzero((labels == c) & correct) for c in classes]
        rows = np.concatenate([r for r in rows if len(r)]) if len(classes) else np.array([], dtype=int)
        if len(rows) == 0:
            splitting.append({})
            models.append(None)
            for key in info:
                info[key].append(None)
            continue
        z, selected, n_components, fitted = project_feature(codes[rows], labels[rows], ranges)
        assignment, k, mixture = cluster_classes(z, labels[rows])
        splitting.append(invert_assignment(assignment))
        # invert_assignment renumbers the surviving modes; keep that map so a per-sample mode can
        # be expressed in the same slot numbering.
        survivors = sorted({int(m) for modes in assignment.values() for m in modes})
        models.append({**fitted, **mixture, "remap": {m: i for i, m in enumerate(survivors)},
                       "rows": rows})
        for key, val in zip(info, (selected, n_components, k, len(rows))):
            info[key].append(val)
        if verbose:
            print(f"feature {f:3d}: {len(classes):3d} classes, {len(rows):5d} rows, layers 0..{selected}, "
                  f"PCA {n_components:3d}, modes {k}, meanings {len(set().union(*map(set, splitting[-1].values())))}")
    return MeaningSplit(splitting, info, models if keep_models else None)


def sample_mode_slots(split, codes, models=None):
    """``[N, F]`` int: each sample's most likely surviving mixture mode, per feature.

    This is the per-sample alternative to paper eq. 8: the concept of a feature is read off the
    sample's own activation code through the fitted transform and mixture, with no class-level
    constraint. Modes that no class was assigned to are not slots of the split, so the argmax
    runs over the surviving ones. Returns slot indices local to the feature (0 for unsplit).
    """
    models = models if models is not None else getattr(split, "models", None)
    if models is None:
        raise ValueError("no fitted models; recompute the split with keep_models=True")
    codes = np.asarray(codes)
    out = np.zeros((len(codes), split.n_feats_init), dtype=int)
    for f, m in enumerate(models):
        if m is None or split.n_meanings[f] == 1:
            continue
        probas = mode_probabilities(m, project_with(m, codes))
        survivors = np.array(sorted(m["remap"]))
        best = survivors[probas[:, survivors].argmax(axis=1)]
        out[:, f] = [m["remap"][int(b)] for b in best]
    return out


def slots_to_masks(split, slots):
    """``[N, F]`` per-feature slot indices -> ``[N, total]`` boolean routing masks."""
    slots = np.asarray(slots)
    masks = np.zeros((len(slots), split.total_features), dtype=bool)
    masks[np.arange(len(slots))[:, None], split.first_slot[None, :] + slots] = True
    return masks
