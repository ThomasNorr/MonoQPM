"""Conformal efficiency of a QPM before and after MonoQPM disentanglement.

Reports Observed Fuzziness (paper eq. 4) for THR and APS. It is threshold-free: no prediction set
is formed, so the number does not depend on a target coverage. Lower is better.

The split is computed by `disentangle_qpm.py` (and reused from its cache if present), then eq. 8
and eq. 9 turn the frozen QPM's features into MonoQPM's logits. Accuracy is reported alongside:
MonoQPM provably cannot change the top-1 prediction, and the printed `top-1 kept` column is the
empirical check of that.

`--rules all` adds the assignment ablations, which vary the two independent choices of the
method: who owns a concept (eq. 14's threshold, or every concept a class's own training images
land in) and how a test image is routed (eq. 8's class logits, or the image's own mixture mode).
Only rules that keep eq. 8 preserve the top-1 guarantee.

Example:

    python evaluate_conformal.py --dataset CUB2011 --arch resnet50 --seed 123456 --cal_frac 0.3333
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from monoqpm.conformal import class_slots_from_samples, fuzziness, mono_logits
from monoqpm.extract import (check_linear_reconstruction, class_assignment, compute_or_load_split,
                             load_qpm, run_folder)
from monoqpm.meaning_split import sample_mode_slots, slots_to_masks

PAPER_RULES = ("QPM", "MonoQPM")
ABLATION_RULES = ("MonoQPM re-routed", "eq. 14 + sample-mode", "free + eq. 8", "free + sample-mode")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="CUB2011",
                   choices=["CUB2011", "ImageNet", "TravelingBirds", "StanfordCars"])
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--seed", default=None, type=int)
    p.add_argument("--model_type", default="qpm", choices=["qpm", "chiqpm", "qsenn", "sldd"])
    p.add_argument("--n_features", default=50, type=int)
    p.add_argument("--n_per_class", default=5, type=int)
    p.add_argument("--cropGT", default=True, type=bool, help='pass "" to disable')
    p.add_argument("--img_size", default=224, type=int)
    p.add_argument("--reduced_strides", default=True, type=bool)
    p.add_argument("--folder", default=None)
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--out", default=None, help="default: <run folder>/monoqpm")
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--rules", default="paper", choices=["paper", "all"],
                   help="'all' adds the assignment ablations")
    p.add_argument("--cal_size", default=10, type=int, help="calibration images per class")
    p.add_argument("--cal_frac", default=None, type=float,
                   help="calibrate on this fraction of each class instead of a fixed count; "
                        "portable across datasets, unlike a fixed count (implies stratified)")
    p.add_argument("--stratified", action="store_true",
                   help="stratify the calibration set by class; the original index arithmetic "
                        "only does so on a balanced, class-ordered test set")
    return p.parse_args()


def iterate_routing(split, test, weight, max_iter=10):
    """Ablation: re-run eq. 8 on the logits eq. 9 produced, until the routing stops moving."""
    logits = np.asarray(test["logits"])
    masks = split.route_masks(logits)
    first, n_iter = None, 0
    for n_iter in range(1, max_iter + 1):
        logits = mono_logits(split, test["features"], logits, weight, masks=masks)
        new_masks = split.route_masks(logits)
        if first is None:
            first = float((new_masks != masks).any(axis=1).mean())
        if np.array_equal(new_masks, masks):
            break
        masks = new_masks
    return logits, n_iter, first


def evaluate(split, train, test, weight, n_classes, rules, fz):
    """Observed Fuzziness of the unsplit QPM and of each assignment rule."""
    out = {"QPM": {**fuzziness(test["logits"], test["labels"], n_classes, **fz),
                   "acc": float((np.asarray(test["logits"]).argmax(1) == test["labels"]).mean())}}
    variants, extra, iterated = {"MonoQPM": dict()}, {}, None
    if "MonoQPM re-routed" in rules:
        # Needs the fitted transforms and the test ReLU codes; both are cached by the caller.
        test_slots = sample_mode_slots(split, test["codes"])
        train_slots = sample_mode_slots(split, train["codes"])
        free_slots = class_slots_from_samples(split, train_slots, train["labels"], split.models)
        sample_masks = slots_to_masks(split, test_slots)
        variants.update({"eq. 14 + sample-mode": dict(masks=sample_masks),
                         "free + eq. 8": dict(class_slots=free_slots),
                         "free + sample-mode": dict(masks=sample_masks, class_slots=free_slots)})
        iterated, n_iter, first = iterate_routing(split, test, weight)
        variants["MonoQPM re-routed"] = None
        extra = {"reroute_iterations": n_iter, "reroute_images_changed": first,
                 "routing_agreement": float(split.route_masks(test["logits"])[sample_masks].mean())}
    for name, kwargs in variants.items():
        logits = iterated if kwargs is None else mono_logits(
            split, test["features"], test["logits"], weight, **kwargs)
        pred = np.asarray(logits).argmax(1)
        out[name] = {**fuzziness(logits, test["labels"], n_classes, **fz),
                     "acc": float((pred == test["labels"]).mean()),
                     "top1_kept": float((pred == np.asarray(test["logits"]).argmax(1)).mean())}
    return {**out, **extra}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = run_folder(args.dataset, args.arch, args.seed, args.cropGT, args.folder)
    out = Path(args.out).expanduser() if args.out else folder / "monoqpm"
    tag = f"{args.model_type}_{args.n_features}_{args.n_per_class}"
    rules = PAPER_RULES + (ABLATION_RULES if args.rules == "all" else ())

    model, _ = load_qpm(args.dataset, args.arch, args.seed, args.model_type, args.n_features,
                        args.n_per_class, args.cropGT, args.img_size, args.reduced_strides,
                        folder, device)
    split, train, test = compute_or_load_split(
        model, args.dataset, out, device, crop=args.cropGT, img_size=args.img_size,
        batch_size=args.batch_size, tag=tag, recompute=args.recompute,
        test_codes=args.rules == "all", keep_models=args.rules == "all")

    weight = class_assignment(model)
    check_linear_reconstruction(model, test["features"], test["logits"])
    n_classes = int(weight.shape[0])
    fz = dict(cal_size=args.cal_size, stratified=args.stratified, cal_frac=args.cal_frac)
    scores = evaluate(split, train, test, weight, n_classes, rules, fz)

    assignment = weight != 0
    print(f"\n{split.n_feats_init} features -> {split.total_features} concepts "
          f"({split.n_meanings.mean():.2f} per feature, {(split.n_meanings > 1).sum()} split); "
          f"{assignment.sum(1).mean():.1f} features/class, {assignment.sum(0).mean():.2f} classes/feature")
    print("\n| rule | THR | APS | acc | top-1 kept |")
    print("|---|---|---|---|---|")
    for rule in rules:
        s = scores[rule]
        kept = "-" if rule == "QPM" else f"{s['top1_kept']:.4f}"
        print(f"| {rule} | {s['THR']:.3f} | {s['APS']:.3f} | {s['acc']:.4f} | {kept} |")

    result = {**scores, "dataset": args.dataset, "arch": args.arch, "seed": args.seed,
              "n_classes": n_classes, "features": int(assignment.shape[1]),
              "concepts": int(split.total_features),
              "concepts_per_feature": float(split.n_meanings.mean()),
              "features_split": int((split.n_meanings > 1).sum()),
              "features_per_class": float(assignment.sum(1).mean()),
              "classes_per_feature": float(assignment.sum(0).mean())}
    with open(out / f"{tag}_conformal.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nwrote", out / f"{tag}_conformal.json")


if __name__ == "__main__":
    main()
