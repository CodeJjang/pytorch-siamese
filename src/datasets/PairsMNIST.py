import torch
from torchvision.datasets import MNIST
from PIL import Image
import numpy as np
from collections import Counter
from tqdm import tqdm


class PairsMNIST(MNIST):
    def __init__(self, **kwargs):
        super(PairsMNIST, self).__init__(**kwargs)
        self._split_mnist_to_pairs()

    def _split_mnist_to_pairs(self):
        # Convert to numpy so we can delete
        self.data = self.data.numpy()
        self.targets = self.targets.numpy()
        new_dataset = []
        new_targets = []

        # Track class histogram to know how many pairs more we can create
        self.original_class_hist = Counter(self.targets)

        self.classes = ['same', 'not_same']
        self.cls_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Init pbar data
        pbar = tqdm(total=int(self._sum_class_hist() / 2), desc='Creating MNIST same/not same pairs')

        while not self._class_hist_is_empty():
            different_label_indices = self._pick_different_label_samples(self.targets)
            if different_label_indices is not None:
                self._update_data(new_dataset, new_targets, different_label_indices, self.cls_to_idx['not_same'])
                pbar.update(1)

            same_label_indices = self._pick_same_label_samples(self.targets)
            if same_label_indices is not None:
                self._update_data(new_dataset, new_targets, same_label_indices, self.cls_to_idx['same'])
                pbar.update(1)

        pbar.close()
        # Update data and targets with new ones and convert to tensors
        self.data = torch.stack(new_dataset)  # New dataset of size (N, 2, 28, 28)
        self.targets = torch.from_numpy(np.array(new_targets))  # New targets of size (N)

    def _update_data(self, new_dataset, new_targets, indices, label_idx):
        new_dataset.append(torch.stack([
            torch.from_numpy(self.data[indices[0]]),
            torch.from_numpy(self.data[indices[1]])
        ]))
        new_targets.append(label_idx)
        self.data = np.delete(self.data, indices, axis=0)
        self.original_class_hist[self.targets[indices[0]]] -= 1
        self.original_class_hist[self.targets[indices[1]]] -= 1
        self.targets = np.delete(self.targets, indices, axis=0)

    def _sum_class_hist(self):
        return sum(self.original_class_hist.values())

    def _class_hist_is_empty(self):
        return self._sum_class_hist() <= 1

    def _gen_reverse_data_index(self, data, targets):
        self.reverse_data_index = {}
        for sample, label in zip(data, targets):
            if not self.reverse_data_index[label]:
                self.reverse_data_index[label] = []
            self.reverse_data_index[label].append(sample)

    def _has_min_amount_of_classes(self, min_amount):
        amount_positive_classes = 0
        for cls in self.original_class_hist.keys():
            if self.original_class_hist[cls] >= min_amount:
                amount_positive_classes += 1
        return amount_positive_classes

    def _class_hist_has_different_labels(self):
        """Checks we have at least two classes each one with 1 instance
        """
        amount_positive_classes = self._has_min_amount_of_classes(1)
        return amount_positive_classes >= 2

    def _class_hist_has_same_labels(self):
        """Checks we have at least one class with two instances
        """
        amount_positive_classes = self._has_min_amount_of_classes(2)
        return amount_positive_classes >= 1

    def _pick_different_label_samples(self, targets):
        if self._class_hist_has_different_labels():
            condition = lambda indices: targets[indices[0]] != targets[indices[1]]
            return self._pick_label_samples(len(targets), condition)

    def _pick_same_label_samples(self, targets):
        if self._class_hist_has_same_labels():
            condition = lambda indices: targets[indices[0]] == targets[indices[1]]
            return self._pick_label_samples(len(targets), condition)

    def _pick_label_samples(self, data_len, condition):
        indices = np.random.choice(data_len, 2, replace=False)
        while not condition(indices):
            indices = np.random.choice(data_len, 2, replace=False)
        return indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        (img1, img2), target = self.data[index], int(self.targets[index])

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, target
