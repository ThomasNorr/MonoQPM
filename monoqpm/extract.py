"""Feed a trained QPM of this repository into the MonoQPM split.

One deterministic pass over a split collects everything the pipeline needs: the ReLU codes
(eq. 15), the head's normalized features and its class logits. The training pass is the expensive
one and is cached, so recomputing a split or re-scoring it costs seconds.
"""
from pathlib import Path

import numpy as np
import torch

from evaluation.load_model import load_model
from get_data import get_data
from monoqpm.meaning_split import extract_relu_codes


def run_folder(dataset, arch, seed, crop=True, folder=None):
    """Where `main.py` writes a run, mirroring `evaluation.load_model.load_model`."""
    if folder is not None:
        return Path(folder).expanduser()
    return Path.home() / f"tmp/{arch}/{dataset}{'_crop' if crop else ''}/{seed}/"


def load_qpm(dataset, arch, seed, model_type="qpm", n_features=50, n_per_class=5, crop=True,
             img_size=224, reduced_strides=True, folder=None, device="cpu"):
    """A finetuned sparse head (`qpm`, `chiqpm`, `qsenn`, `sldd`), ready for evaluation."""
    model, folder = load_model(dataset, arch, seed, model_type, crop, n_features, n_per_class,
                               img_size, reduced_strides, folder)
    return model.to(device).eval(), folder


def class_assignment(model):
    """``W [n_classes, n_features]``: nonzero where a class is assigned a feature."""
    return model.linear.weight.detach().cpu().numpy()


def check_linear_reconstruction(model, features, logits_ref=None, tol=1e-2):
    """The head must be linear in the extracted features; returns the logits it implies.

    Eq. 9 rebuilds the logits from the split features, so it is only valid if the original logits
    are ``features @ W.T``. QPM has no bias, which this asserts.
    """
    w = class_assignment(model)
    rebuilt = np.asarray(features, dtype=np.float64) @ w.T.astype(np.float64)
    bias = getattr(model.linear.layer, "bias", None)
    assert bias is None or float(bias.detach().abs().max()) == 0, "eq. 9 assumes a bias-free head"
    if logits_ref is not None:
        dev = float(np.abs(rebuilt - np.asarray(logits_ref, dtype=np.float64)).max())
        assert dev < tol, f"features @ W.T deviates from the model's own logits by {dev}"
    return rebuilt


def extract(model, dataset, key, device, crop=True, img_size=224, batch_size=64, with_codes=True):
    """One pass over the train or test split, in dataset order, with the evaluation transform."""
    train_loader, test_loader = get_data(dataset, crop=crop, img_size=img_size)
    if key == "Train":
        data = train_loader.dataset
        data.transform = test_loader.dataset.transform  # no augmentation, so row r is sample r
    else:
        data = test_loader.dataset
    return extract_relu_codes(model, data, device, batch_size=batch_size,
                              num_workers=train_loader.num_workers, with_codes=with_codes)


def compute_or_load_split(model, dataset, out_dir, device, crop=True, img_size=224, batch_size=64,
                          tag="qpm", recompute=False, verbose=True, test_codes=False,
                          keep_models=False):
    """``(split, train, test)`` for one checkpoint, caching both passes and the split.

    ``test_codes`` and ``keep_models`` are only needed by the per-image ablation of eq. 8; a cache
    written without them is recomputed when they are asked for.
    """
    from monoqpm.meaning_split import MeaningSplit, compute_meaning_split
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    passes = {}
    for key, with_codes in (("Train", True), ("Test", test_codes)):
        path = out_dir / f"{tag}_{key.lower()}.npz"
        cached = None
        if path.exists() and not recompute:
            cached = dict(np.load(path, allow_pickle=True))
            if with_codes and "codes" not in cached:
                cached = None
        if cached is not None:
            cached["ranges"] = [tuple(r) for r in cached["ranges"]] if cached["ranges"].size else None
            cached.setdefault("codes", None)
            passes[key] = cached
            if verbose:
                print(f"  loaded cached {key} pass from {path.name}")
            continue
        passes[key] = extract(model, dataset, key, device, crop, img_size, batch_size, with_codes)
        np.savez(path, ranges=np.array(passes[key]["ranges"] or []),
                 **{k: v for k, v in passes[key].items() if k != "ranges" and v is not None})
        if verbose:
            print(f"  {key}: {len(passes[key]['labels'])} images"
                  + (f", ReLU codes {passes[key]['codes'].shape}" if with_codes else ""), flush=True)
    train, test = passes["Train"], passes["Test"]
    split_path = out_dir / f"{tag}_split.pkl"
    split = None
    if split_path.exists() and not recompute:
        split = MeaningSplit.load(split_path)
        if keep_models and split.models is None:
            split = None
        elif verbose:
            print(f"  loaded cached split from {split_path.name}")
    if split is None:
        split = compute_meaning_split(train["codes"], train["ranges"], train["logits"],
                                      train["labels"], class_assignment(model), verbose=verbose,
                                      keep_models=keep_models)
        split.save(split_path)
    return split, train, test
