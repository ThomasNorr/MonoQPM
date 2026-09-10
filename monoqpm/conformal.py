"""Observed Fuzziness of THR / APS conformal prediction, and MonoQPM logits (paper eq. 9).

Scores come from torchcp's ``THR()`` / ``APS()`` with their defaults, and the p-value of label
``y`` is ``(#{calibration scores >= s(x, y)} + 1) / (n_cal + 1)``. Observed Fuzziness of a test
image is the sum of the p-values of its FALSE labels; the reported number is its mean over the
non-calibration test images (paper eq. 4). It is threshold-free: no prediction set is formed, so
it does not depend on a target coverage.
"""
import numpy as np
import torch


def calibration_split(n_test, n_classes, cal_size=10, labels=None, cal_frac=None, seed=0):
    """Calibration/evaluation indices.

    Without ``labels`` this is the original index arithmetic, ``i % (n_test // n_classes) <
    cal_size``, which stratifies only because a class-ordered test set with near-equal class sizes
    makes every such block one class (true for CUB, false for an imbalanced set such as
    HAM10000, where it drew 0 images of one class and 51 of another).

    With ``labels`` the split is stratified per class and drawn with a seeded RNG. ``cal_frac``
    takes that fraction of each class, which is the portable choice: a fixed count per class is
    34.5% of CUB's test set but only 3.5% of HAM10000's, so the p-value resolution (1 / (n_cal+1))
    would differ by an order of magnitude between datasets. ``cal_size`` takes a fixed count.
    """
    idx = np.arange(n_test)
    if labels is None:
        per_class = int(np.ceil(n_test // n_classes))
        cal = idx[(idx % per_class) < cal_size]
        keep = idx[(idx % per_class) >= cal_size]
        assert np.floor(len(cal) / n_classes) == cal_size
        return cal, keep
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    cal = []
    for c in np.unique(labels):
        rows = np.flatnonzero(labels == c)
        k = max(1, int(round(cal_frac * len(rows)))) if cal_frac is not None else cal_size
        cal.append(rng.choice(rows, size=min(k, len(rows)), replace=False))
    cal = np.sort(np.concatenate(cal))
    return cal, np.setdiff1d(idx, cal)


def observed_fuzziness(cal_scores, test_scores, true_scores):
    """Mean over test samples of the sum of false-label p-values (non-conformity scores: larger = worse)."""
    cal_sorted, _ = torch.sort(cal_scores)
    n_cal = len(cal_sorted)
    p_all = (n_cal - torch.bucketize(test_scores.flatten(), cal_sorted) + 1) / (n_cal + 1)
    p_all = p_all.view(test_scores.shape).sum(dim=1)
    p_true = (n_cal - torch.bucketize(true_scores, cal_sorted) + 1) / (n_cal + 1)
    return float((p_all - p_true).mean())


def fuzziness(logits, labels, n_classes, scores=("THR", "APS"), cal_size=10, seed=0,
              stratified=False, cal_frac=None):
    """``{score: observed fuzziness}`` for the given logits ``[N, C]`` and labels ``[N]``."""
    from torchcp.classification.scores import APS, THR
    logits = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    strat = stratified or cal_frac is not None
    cal, keep = calibration_split(len(logits), n_classes, cal_size,
                                  np.asarray(labels) if strat else None, cal_frac, seed)
    out = {}
    with torch.no_grad():
        for name in scores:
            torch.manual_seed(seed)  # APS is randomized in torchcp
            fn = {"THR": THR, "APS": APS}[name]()
            cal_scores = fn(logits[cal], labels[cal])
            test_scores = fn(logits[keep])
            true_scores = fn(logits[keep], labels[keep])
            out[name] = observed_fuzziness(cal_scores, test_scores, true_scores)
    return out


def concept_slots_per_class(split, f):
    """``{class: sorted slot indices}`` of feature ``f`` (a class may sit in several concepts)."""
    per_class = {}
    for classes, modes in split.splitting[f].items():
        for c in classes:
            per_class.setdefault(int(c), set()).update(int(split.first_slot[f] + m) for m in modes)
    return {c: tuple(sorted(s)) for c, s in per_class.items()}


def class_slots_from_samples(split, train_slots, labels, models):
    """``[{class: slots}]`` per feature, taken from where the TRAINING samples actually route.

    The per-sample alternative to paper eq. 14: a class owns every concept at least one of its
    own (correctly classified) training samples falls into, with no homogeneity threshold.
    """
    labels = np.asarray(labels)
    out = []
    for f, m in enumerate(models):
        if m is None:
            out.append({})
            continue
        rows, base = m["rows"], int(split.first_slot[f])
        slots, lab = np.asarray(train_slots)[rows, f], labels[rows]
        out.append({int(c): tuple(base + np.unique(slots[lab == c])) for c in np.unique(lab)})
    return out


def mono_logits(split, features, logits, weight, inactive="min", train_features=None,
                masks=None, class_slots=None):
    """MonoQPM logits, paper eq. 9: per row, sum over its features of the max over its concepts.

    ``weight`` is the QPM's class/feature assignment ``W [C, F]``; the result is ``[N, C]``.

    ``inactive`` sets ``mu_d`` of paper eq. 8, the value of a concept slot the image is NOT routed
    to: ``"min"`` (the default, and what the published numbers use) is the per-feature minimum
    over the evaluated set; ``"train_min"`` / ``"train_mean"`` take a constant from
    ``train_features`` instead; ``"zero"`` sets 0; ``"min0"`` uses the per-image ``min(0, f_d)``.
    This choice matters more than any other: on a CUB200 QPM roughly 82% of feature values are
    negative, and ``"zero"`` is worse than not splitting at all.

    ``masks`` overrides the eq. 8 routing (e.g. each sample's own mixture mode), and
    ``class_slots`` overrides which concepts a class reads (e.g. derived from where its training
    samples route rather than from the eq. 14 threshold).
    """
    features = np.asarray(features, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if weight.min() < 0:
        # max_c (f_c * w) = w * max_c f_c only for w >= 0, which both heads satisfy (binary).
        raise NotImplementedError("negative assignment weights would flip the max over concepts")
    masks = split.route_masks(np.asarray(logits)) if masks is None else np.asarray(masks)
    gathered = features[:, split.slot_to_orig]
    if inactive == "min0":
        base = np.minimum(gathered, 0)
    elif inactive == "zero":
        base = np.zeros_like(gathered)
    elif inactive in ("min", "train_min", "train_mean"):
        stat = {"min": lambda: features.min(axis=0), "train_min": lambda: train_features.min(axis=0),
                "train_mean": lambda: train_features.mean(axis=0)}[inactive]()
        base = np.broadcast_to(np.asarray(stat, dtype=np.float64)[split.slot_to_orig], gathered.shape).copy()
    else:
        raise ValueError(inactive)
    slots = np.where(masks, gathered, base)  # [N, total]

    row_class = np.arange(weight.shape[0])
    out = np.zeros((len(features), weight.shape[0]))
    for f in range(split.n_feats_init):
        per_class = concept_slots_per_class(split, f) if class_slots is None else class_slots[f]
        uses_f = weight[:, f] != 0
        # Rows sharing a class share their concept set, so group by it and take one max per group.
        groups = {}
        for c, s in per_class.items():
            groups.setdefault(tuple(s), []).append(c)
        covered = np.zeros(len(weight), dtype=bool)
        for slot_ids, classes in groups.items():
            rows = np.flatnonzero(uses_f & np.isin(row_class, classes))
            if not len(rows):
                continue
            covered[rows] = True
            best = slots[:, list(slot_ids)].max(axis=1)  # [N]
            out[:, rows] += best[:, None] * weight[rows, f][None, :]
        # A class that uses the feature but owns no concept of it (every training image of it is
        # misclassified, so it never entered the clustering) can only see the baseline.
        rest = np.flatnonzero(uses_f & ~covered)
        if len(rest):
            out[:, rest] += base[:, split.first_slot[f]][:, None] * weight[rest, f][None, :]
    return out
