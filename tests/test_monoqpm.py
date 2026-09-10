"""Tests for the MonoQPM pipeline. Run from the repository root:

    python -m pytest tests/test_monoqpm.py        # or simply: python tests/test_monoqpm.py

They cover the three parts that are easy to get subtly wrong: the Fisher-Jenks breaks (checked
against brute-force optimality, and against `jenkspy` if it happens to be installed), eq. 8/9
(checked against a hand-written reference), and the ReLU hook layout of eq. 15.
"""
import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monoqpm.conformal import (calibration_split, class_slots_from_samples,  # noqa: E402
                               concept_slots_per_class, mono_logits, observed_fuzziness)
from monoqpm.meaning_split import (MeaningSplit, compute_meaning_split, jenks_breaks,  # noqa: E402
                                   mode_order, participation_ratio, project_with,
                                   relu_activation_fractions, sample_mode_slots, select_layers,
                                   slots_to_masks)


def _toy_split():
    """A 6-class, 4-feature toy problem whose feature 0 carries two concepts."""
    C, F = 6, 4
    W = np.zeros((C, F))
    W[:, 0] = 1
    W[:, 1] = 1
    W[[0, 3], 2] = 1
    W[[1, 4, 5], 3] = 1
    splitting = [{(0, 1, 2): np.array([0]), (3, 4, 5): np.array([1])},
                 {tuple(range(C)): np.array([0])},
                 {(0, 3): np.array([0])},
                 {(1, 4, 5): np.array([0])}]
    return MeaningSplit(splitting), W


def test_jenks_matches_brute_force():
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = int(rng.integers(5, 13))
        v = np.round(rng.normal(size=n) * 3, 1)
        if trial % 3 == 0:
            v[: n // 2] = np.round(v[: n // 2])  # force duplicates
        for k in range(1, min(4, len(np.unique(v))) + 1):
            breaks = jenks_breaks(v, k)
            s = np.sort(v)
            cuts = [int(np.searchsorted(s, breaks[i], side="right")) for i in range(1, k)]
            sse = sum(((p - p.mean()) ** 2).sum() for p in np.split(s, cuts))
            best = min(sum(((p - p.mean()) ** 2).sum() for p in np.split(np.sort(v), c))
                       for c in itertools.combinations(range(1, len(v)), k - 1)) if k > 1 else sse
            assert np.isclose(sse, best), (v, k, breaks)
    # the documented example from jenkspy
    assert jenks_breaks([1.3, 7.1, 7.3, 2.3, 3.9, 4.1, 7.8, 1.2, 4.3, 7.3, 5.0, 4.3], 3) == \
        [1.2, 2.3, 5.0, 7.8]


def test_jenks_matches_jenkspy_if_available():
    try:
        import jenkspy
    except ImportError:
        return  # optional: the point of the reimplementation is not to need it
    rng = np.random.default_rng(1)
    for n in (50, 300):
        v = rng.normal(size=n)
        v[:5] = v[5:10]
        for k in (1, 2, 5, 12):
            assert np.allclose(jenks_breaks(v, k), jenkspy.jenks_breaks(v, k))


def test_participation_ratio_and_depth_selection():
    assert np.isclose(participation_ratio(np.zeros((4, 3))), 1.0)
    rng = np.random.default_rng(1)
    ranges = [(0, 4), (4, 12), (12, 28), (28, 60), (60, 70)]
    codes = np.zeros((200, 70))
    codes[:, 0:4] = rng.normal(size=(200, 4)) @ rng.normal(size=(4, 4))
    codes[:, 4:12] = rng.normal(size=(200, 8))                              # local max
    codes[:, 12:28] = rng.normal(size=(200, 2)) @ rng.normal(size=(2, 16))
    codes[:, 28:60] = rng.normal(size=(200, 32))                            # global max
    codes[:, 60:70] = rng.normal(size=(200, 10))
    selected, _ = select_layers(codes, ranges)
    assert selected == 1, "eq. 17 takes the last maximum strictly before the global one"


def test_mode_order_is_the_greedy_path():
    assert mode_order(np.array([0.0])).tolist() == [0]
    assert mode_order(np.array([0.0, 10.0, 1.0, 9.0])).tolist() == list(np.argsort([0, 2, 3, 1]))


def test_eq8_eq9_against_a_hand_written_reference():
    split, W = _toy_split()
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(50, 4))
    logits = feats @ W.T
    mono = mono_logits(split, feats, logits, W)

    mu = feats.min(axis=0)
    ref = np.zeros_like(logits)
    for n in range(len(feats)):
        for y in range(W.shape[0]):
            for f in np.flatnonzero(W[y]):
                if f == 0:
                    a, b = logits[n, [0, 1, 2]].max(), logits[n, [3, 4, 5]].max()
                    mine = a if y < 3 else b
                    ref[n, y] += feats[n, 0] if mine == max(a, b) else mu[0]
                else:
                    ref[n, y] += feats[n, f]
    assert np.allclose(mono, ref)
    assert (mono <= logits + 1e-9).all(), "a concept slot can only lose a class its activation"


def test_top1_is_preserved_by_construction():
    split, W = _toy_split()
    rng = np.random.default_rng(3)
    for _ in range(5):
        feats = rng.normal(size=(200, 4))
        logits = feats @ W.T
        mono = mono_logits(split, feats, logits, W)
        assert (mono.argmax(1) == logits.argmax(1)).all()


def test_unsplit_features_are_untouched():
    split, W = _toy_split()
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(20, 4))
    logits = feats @ W.T
    flat = MeaningSplit([{tuple(np.flatnonzero(W[:, f])): np.array([0])} for f in range(4)])
    assert np.allclose(mono_logits(flat, feats, logits, W), logits)


def test_alternative_routing_and_ownership():
    split, W = _toy_split()
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(50, 4))
    logits = feats @ W.T
    mu = feats.min(axis=0)
    slots = np.stack([rng.integers(0, split.n_meanings[f], size=50) for f in range(4)], axis=1)
    masks = slots_to_masks(split, slots)
    for f in range(4):
        assert (masks[:, split.slot_to_orig == f].sum(1) == 1).all()
    assert np.allclose(mono_logits(split, feats, logits, W),
                       mono_logits(split, feats, logits, W, masks=split.route_masks(logits)))
    mono = mono_logits(split, feats, logits, W, masks=masks)
    ref = np.zeros_like(logits)
    for n in range(50):
        for y in range(6):
            for f in np.flatnonzero(W[y]):
                routed = int(split.first_slot[f] + slots[n, f])
                ref[n, y] += feats[n, f] if routed in concept_slots_per_class(split, f)[y] else mu[f]
    assert np.allclose(mono, ref)
    # owning every concept of a feature is degenerate: the logits come back unchanged
    everything = [{c: tuple(range(split.first_slot[f], split.first_slot[f] + split.n_meanings[f]))
                   for c in np.flatnonzero(W[:, f])} for f in range(4)]
    assert np.allclose(mono_logits(split, feats, logits, W, masks=masks, class_slots=everything),
                       logits)


def test_end_to_end_recovers_a_planted_split():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(6), 40)
    D, ranges = 40, [(0, 10), (10, 25), (25, 40)]
    codes = rng.normal(size=(len(labels), D)) * 0.1
    proto = rng.normal(size=(2, D))
    codes[labels < 3] += proto[0]
    codes[labels >= 3] += proto[1]
    codes = (codes > 0).astype(np.float32) + rng.normal(size=codes.shape).astype(np.float32) * 0.01
    logits = np.eye(6)[labels] * 5 + rng.normal(size=(len(labels), 6)) * 0.5
    W = np.zeros((6, 3))
    W[:, 0] = 1
    W[[0, 1, 2], 1] = 1
    W[[0, 3], 2] = 1
    split = compute_meaning_split(codes, ranges, logits, labels, W, verbose=False, keep_models=True)
    assert split.n_meanings[0] >= 2, "the planted two-group feature must split"
    assert split.n_meanings[2] == 1, "a two-class feature cannot split (k_max = 1)"
    # eq. 14 lets a class own several concepts of one feature, so the row sums may GROW; what
    # must hold is that every assigned (class, feature) keeps its weight in at least one concept
    # and that no unassigned one gains any.
    sw = split.split_weight(W)
    for c in range(W.shape[0]):
        for f in range(W.shape[1]):
            got = sw[c, split.slot_to_orig == f]
            assert (got == W[c, f]).any() if W[c, f] else not got.any()

    slots = sample_mode_slots(split, codes)
    assert (slots[:, 2] == 0).all()
    # The per-image rule may scatter a group over several concepts (that is the whole point of
    # it), but the two planted groups must not share one.
    first, second = set(slots[labels < 3, 0]), set(slots[labels >= 3, 0])
    assert not (first & second), (first, second)
    # Free ownership must be exactly the set of concepts a class's own training images land in,
    # expressed in GLOBAL slot indices.
    free = class_slots_from_samples(split, slots, labels, split.models)
    for f, model in enumerate(split.models):
        if model is None:
            continue
        rows, lab = model["rows"], labels[model["rows"]]
        for c in np.unique(lab):
            want = tuple(sorted(split.first_slot[f] + np.unique(slots[rows][lab == c, f])))
            assert tuple(free[f][int(c)]) == want, (f, c, free[f][int(c)], want)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_split.pkl")
    try:
        split.save(path)
        back = MeaningSplit.load(path)
        assert np.array_equal(sample_mode_slots(back, codes), slots)
        assert np.allclose(project_with(back.models[0], codes[:3]), project_with(split.models[0], codes[:3]))
    finally:
        os.path.exists(path) and os.remove(path)


def test_calibration_split_and_fuzziness():
    labels = np.repeat(np.arange(4), 25)
    cal, keep = calibration_split(len(labels), 4, cal_size=5, labels=labels)
    assert len(cal) == 20 and np.bincount(labels[cal]).tolist() == [5, 5, 5, 5]
    assert not set(cal) & set(keep)
    cal, keep = calibration_split(len(labels), 4, labels=labels, cal_frac=0.4)
    assert np.bincount(labels[cal]).tolist() == [10, 10, 10, 10]
    # a perfectly calibrated, uninformative predictor gives every false label a high p-value
    scores = torch.zeros(30)
    assert 0.0 <= observed_fuzziness(scores, torch.zeros(10, 4), torch.zeros(10)) <= 4.0


def test_relu_hook_layout_on_resnet18():
    from architectures.model_mapping import get_model
    model = get_model("resnet18", 200, True).eval()
    x = torch.randn(2, 3, 224, 224)
    codes, ranges, out = relu_activation_fractions(model, x)
    widths = [e - s for s, e in ranges]
    # BasicBlock fires its ReLU twice; layer4's last one is applied outside the block
    assert widths == [64] + [64] * 4 + [128] * 4 + [256] * 4 + [512] * 3
    assert codes.shape == (2, 3392) and out.shape == (2, 200)
    assert float(codes.min()) >= 0 and float(codes.max()) <= 1
    with torch.no_grad():
        stem = torch.relu(model.bn1(model.conv1(x)))
    assert torch.allclose(codes[:, :64], (stem > 0).float().flatten(2).mean(2))
    codes2, ranges2, out2 = relu_activation_fractions(model, x, with_final_features=True)
    assert ranges2 == ranges and torch.allclose(codes, codes2)
    assert isinstance(out2, list) and out2[1].shape == (2, 512)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  FAIL {name}: {exc}")
    print("ALL OK" if not failed else f"{failed} test(s) failed")
    sys.exit(1 if failed else 0)
