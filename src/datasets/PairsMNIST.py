from torchvision.datasets import MNIST
from PIL import Image
import numpy as np
from collections import Counter


class PairsMNIST(MNIST):
    def __init__(self, **kwargs):
        super(PairsMNIST, self).__init__(**kwargs)
        new_dataset = []
        new_targets = []
        class_hist = Counter(self.targets)
        self.classes = ['same', 'not_same']
        same_label_idx = 0
        not_same_label_idx = 1
        targets = self.targets
        while not self._class_hist_is_empty(class_hist):
            different_label_indices = self._pick_different_label_samples(targets)
            if different_label_indices:
                self._update_data(new_dataset, new_targets, different_label_indices, not_same_label_idx, class_hist)

            same_label_indices = self._pick_same_label_samples(targets)
            if same_label_indices:
                self._update_data(new_dataset, new_targets, same_label_indices, same_label_idx, class_hist)

    def _update_data(self, new_dataset, new_targets, indices, label_idx, class_hist):
        new_dataset.push(self.data[indices[0]], self.data[indices[1]])
        new_targets.push(label_idx)
        del self.data[indices[0]]
        del self.data[indices[1]]
        class_hist[self.targets[indices[0]]] -= 1
        class_hist[self.targets[indices[1]]] -= 1
        del self.targets[indices[0]]
        del self.targets[indices[1]]

    def _class_hist_is_empty(self, class_hist):
        return sum(class_hist.values()) <= 1

    def _gen_reverse_data_index(self, data, targets):
        self.reverse_data_index = {}
        for sample, label in zip(data, targets):
            if not self.reverse_data_index[label]:
                self.reverse_data_index[label] = []
            self.reverse_data_index[label].append(sample)

    def _has_min_amount_of_classes(self, class_hist, min_amount):
        amount_positive_classes = 0
        for cls in class_hist.keys():
            if class_hist[cls] > min_amount:
                amount_positive_classes += 1
        return amount_positive_classes

    def _class_hist_has_different_labels(self, class_hist):
        """Checks we have at least two classes each one with 1 instance
        """
        amount_positive_classes = self._has_min_amount_of_classes(class_hist, 1)
        return amount_positive_classes >= 2

    def _class_hist_has_same_labels(self, class_hist):
        """Checks we have at least one class with two instances
        """
        amount_positive_classes = self._has_min_amount_of_classes(class_hist, 2)
        return amount_positive_classes >= 1

    def _pick_different_label_samples(self, targets, class_hist):
        if self._class_hist_has_different_labels(class_hist):
            condition = lambda indices: targets[indices[0]] != targets[indices[1]]
            return self._pick_label_samples(len(targets), condition)

    def _pick_same_label_samples(self, targets, class_hist):
        if self._class_hist_has_same_labels(class_hist):
            condition = lambda indices: targets[indices[0]] == targets[indices[1]]
            return self._pick_label_samples(len(targets), condition)

    def _pick_label_samples(self, data_len, condition):
        indices = np.random.choice(data_len, 2, replace=False)
        while not condition(indices):
            indices = np.random.choice(data_len, 2, replace=False)
        return indices

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img1, img2, target = self.data[index], int(self.targets[index])

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, target
