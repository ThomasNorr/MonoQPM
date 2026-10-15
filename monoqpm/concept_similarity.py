"""Ground Truth Concept Classes Similarity (paper eq. 19 with Psi_gt).

For every concept c (a column of the class/concept assignment) with at least two classes, take the
mean of psi_gt(u, v) over its class pairs u < v; the metric is the unweighted mean of that over all
such concepts, pooled across features. Concepts with fewer than two classes have no pair and are
skipped. For the frozen QPM a concept is a feature (a column of W), for MonoQPM a column of the
split assignment ``MeaningSplit.split_weight(W)``.

Psi_gt is
  * CUB2011 / TravelingBirds: the cosine similarity of the classes' 312 continuous attribute
    annotations (``CUB200Class.get_class_sim``),
  * CUBCars: 1 if both classes come from the same source dataset (CUB-2011 or Stanford Cars),
    0 otherwise. A value of 1.0 therefore means no concept mixes birds and cars.
"""
import numpy as np


def cubcars_class_similarity(n_cub=200, n_classes=396):
    sims = np.zeros((n_classes, n_classes))
    sims[:n_cub, :n_cub] = 1
    sims[n_cub:, n_cub:] = 1
    return sims


def gt_class_similarity(dataset):
    """``[C, C]`` Psi_gt, or None for a dataset without ground-truth class similarity."""
    if dataset == "CUBCars":
        return cubcars_class_similarity()
    if dataset in ("CUB2011", "TravelingBirds"):
        from dataset_classes.cub200 import CUB200Class
        return np.asarray(CUB200Class.get_class_sim())
    return None


def concept_classes_similarity(assignment, class_sims):
    """Eq. 19: ``assignment`` is ``[C, n_concepts]``; a class belongs to concept c if its weight is > 0."""
    assignment = np.asarray(assignment)
    class_sims = np.asarray(class_sims)
    per_concept = []
    for c in range(assignment.shape[1]):
        classes = np.flatnonzero(assignment[:, c] > 0)
        if len(classes) < 2:
            continue
        u, v = np.triu_indices(len(classes), k=1)
        per_concept.append(class_sims[classes[u], classes[v]].mean())
    return float(np.mean(per_concept)) if per_concept else 0.0
