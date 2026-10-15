import torch
from tqdm import tqdm


class LazyTestFeatureMaps:
    """Test feature maps of ``rel_features``, recomputed slice by slice instead of buffered.

    Buffering them all needs n_test x n_features x H x W floats: 89 GB for a dense resnet50 on
    CUBCars (13,835 test images, 2048 x 28 x 28), more than the host has. The metric evaluators
    only ever read contiguous slices ``maps[a:b]``, so this recomputes exactly those images from
    the unshuffled test set with the deterministic test transform.
    """

    def __init__(self, model, dataset, rel_features, device):
        self.model, self.dataset, self.rel_features, self.device = model, dataset, rel_features, device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        assert isinstance(index, slice) and index.step in (None, 1), index
        images = torch.stack([self.dataset[i][0] for i in range(*index.indices(len(self)))])
        with torch.no_grad():
            _, feature_maps, _ = self.model(images.to(self.device), with_feature_maps=True,
                                            with_final_features=True)
        return feature_maps[:, self.rel_features].to("cpu")



def get_metrics_for_model(train_loader, test_loader, model, metric_evaluator):
    (features_train, feature_maps_train, outputs_train, features_test, feature_maps_test,
     outputs_test, labels) = [], [], [], [], [], [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model = model.to(device)
    training_transforms = train_loader.dataset.transform
    train_loader.dataset.transform = test_loader.dataset.transform # Use test transform for train
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=100, shuffle=False) # Turn off shuffling
    print("Going in get metrics")
    linear_matrix = model.linear.weight
    entries = torch.nonzero(linear_matrix)
    rel_features = torch.unique(entries[:, 1])
    test_labels = []
    with torch.no_grad():
        iterator = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in iterator:
            xs1 = data.to("cuda")
            output, feature_maps,  final_features = model(xs1,   with_feature_maps=True, with_final_features=True,)
            outputs_train.append(output.to("cpu"))
            features_train.append(final_features.to("cpu"))
            labels.append(target.to("cpu"))
        total = 0
        correct = 0
        iterator = tqdm(enumerate(test_loader), total=len(test_loader))
        for batch_idx, (data, target) in iterator:
            xs1 = data.to("cuda")
            output, feature_maps, final_features = model(xs1, with_feature_maps=True,
                                                         with_final_features=True, )
            features_test.append(final_features.to("cpu"))
            outputs_test.append(output.to("cpu"))
            total += target.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(target.to("cuda")).sum().item()
            test_labels.append(target)
        print("test accuracy: ", correct / total)
        features_train = torch.cat(features_train)
        outputs_train = torch.cat(outputs_train)
        feature_maps_test = LazyTestFeatureMaps(model, test_loader.dataset, rel_features, device)
        outputs_test = torch.cat(outputs_test)
        features_test = torch.cat(features_test)
        labels = torch.cat(labels)
        test_labels = torch.cat(test_labels)
        linear_matrix = linear_matrix[:, rel_features]
    print("Shape of linear matrix: ", linear_matrix.shape)
    all_metrics_dict = metric_evaluator(features_train,  outputs_train,
                                             feature_maps_test,
                                             outputs_test, linear_matrix,  labels, test_labels, features_test)
    result_dict = {"Accuracy": correct / total,  "NFfeatures": linear_matrix.shape[1],
                   "PerClass": torch.nonzero(linear_matrix).shape[0] / linear_matrix.shape[0],
                   }
    result_dict.update(all_metrics_dict)
    print(result_dict)
    # Reset Train transforms
    train_loader.dataset.transform = training_transforms
    return result_dict
