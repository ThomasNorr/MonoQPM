"""Disentangle a trained QPM into MonoQPM: split every polysemantic feature into its concepts.

Runs the pipeline of paper Sections 5.1-5.3 on a checkpoint written by `main.py` and caches the
result. `evaluate_conformal.py` then scores the conformal efficiency of the disentangled model.

Writes to `<run folder>/monoqpm/`:

* `<model>_train.npz` / `<model>_test.npz` -- the deterministic passes (the train one carries the
  ReLU codes of eq. 15, which are the expensive part),
* `<model>_split.pkl` -- the concepts, loadable with `MeaningSplit.load`,
* `<model>_split.json` -- a summary.

Example:

    python disentangle_qpm.py --dataset CUB2011 --arch resnet50 --seed 123456
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from monoqpm.extract import class_assignment, compute_or_load_split, load_qpm, run_folder


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
    p.add_argument("--folder", default=None, help="run folder; derived from the other flags if absent")
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--out", default=None, help="default: <run folder>/monoqpm")
    p.add_argument("--recompute", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = run_folder(args.dataset, args.arch, args.seed, args.cropGT, args.folder)
    out = Path(args.out).expanduser() if args.out else folder / "monoqpm"
    tag = f"{args.model_type}_{args.n_features}_{args.n_per_class}"

    model, _ = load_qpm(args.dataset, args.arch, args.seed, args.model_type, args.n_features,
                        args.n_per_class, args.cropGT, args.img_size, args.reduced_strides,
                        folder, device)
    split, train, test = compute_or_load_split(
        model, args.dataset, out, device, crop=args.cropGT, img_size=args.img_size,
        batch_size=args.batch_size, tag=tag, recompute=args.recompute)

    assignment = class_assignment(model) != 0
    summary = {**split.summary(),
               "features_per_class": float(assignment.sum(1).mean()),
               "classes_per_feature": float(assignment.sum(0).mean()),
               "train_accuracy": float((train["logits"].argmax(1) == train["labels"]).mean()),
               "split_pickle": str(out / f"{tag}_split.pkl")}
    with open(out / f"{tag}_split.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "meanings_per_feature"}, indent=2))

    masks = split.route_masks(test["logits"])
    assert (masks.sum(axis=1) >= split.n_feats_init).all()
    multi = np.mean([(masks[:, split.slot_to_orig == f].sum(1) > 1).mean()
                     for f in range(split.n_feats_init)])
    print(f"{split.n_feats_init} features -> {split.total_features} concepts; (image, feature) "
          f"pairs routed to 2+ concepts (ties in eq. 8): {multi:.4f}")


if __name__ == "__main__":
    main()
