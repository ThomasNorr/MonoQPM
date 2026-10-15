"""
CUBCars: the union of CUB-2011 and Stanford Cars (396 classes).

Classes 0-199 are the CUB-2011 classes, 200-395 the Stanford Cars classes shifted by 200, and the
samples are ordered all CUB images first, then all Cars images, each in its own dataset's order.
This mirrors the dataset the MonoQPM paper was trained on, including the order, which matters
because the conformal calibration split (conformalPrediction/eval_cp.get_logits_and_labels)
selects samples by index.

`crop` applies to the CUB half only, exactly as for CUB2011: the GT-cropped images under
PPCUB200. The Cars images are always used uncropped. One transform is shared by both halves.
"""
import os

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from dataset_classes.cub200 import CUB200Class
from dataset_classes.stanfordcars import StanfordCarsClass


class CUBCarsClass(Dataset):
    name = "CUBCars"
    n_cub_classes = 200

    def __init__(self, train, transform, crop=True):
        self.train = train
        self.transform = transform
        self.crop = crop
        cub = CUB200Class(train, None, crop)
        cars = StanfordCarsClass(train, None, download=False)
        cub_samples = [(os.path.join(cub.root, cub.base_folder, path), target - 1)
                       for path, target in zip(cub.data.filepath, cub.data.target)]
        cars_samples = [(path, target + self.n_cub_classes) for path, target in cars.samples]
        self.samples = cub_samples + cars_samples
        self.targets = [target for _, target in self.samples]
        self.loader = default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path)
        img = self.transform(img)
        return img, target

    def get_cub_cars_class_indices(self):
        return list(range(self.n_cub_classes)), list(range(self.n_cub_classes, 396))
