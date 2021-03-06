import torch
from torchvision.datasets import MNIST
from PIL import Image
import numpy as np
from tqdm import tqdm
import random


class PairsMNIST(MNIST):
    def __init__(self, **kwargs):
        super(PairsMNIST, self).__init__(**kwargs)
        self._split_mnist_to_pairs()

    def _split_mnist_to_pairs(self):
        data = self.data
        classes_indices = [np.where(self.targets == i)[0] for i in range(10)]
        orig_classes_len = len(self.classes)

        x0_data = []
        x1_data = []
        new_targets = []
        x0_original_targets = []
        x1_original_targets = []

        self.original_classes = self.classes
        self.original_class_to_idx = self.class_to_idx
        self.classes = ['not_same', 'same']
        self.cls_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        pairs_per_class_amount = min([len(classes_indices[d]) for d in range(10)]) - 1

        # Init pbar data
        pbar = tqdm(total=int(orig_classes_len * pairs_per_class_amount), desc='Creating MNIST same/not same pairs')

        for cls in range(orig_classes_len):
            for cls_idx in range(pairs_per_class_amount):
                same_label_idx = classes_indices[cls][cls_idx]
                second_same_label_idx = classes_indices[cls][cls_idx + 1]

                # Create a pair of the same label
                x0_data.append(data[same_label_idx])
                x1_data.append(data[second_same_label_idx])
                new_targets.append(self.cls_to_idx['same'])
                x0_original_targets.append(self.targets[same_label_idx])
                x1_original_targets.append(self.targets[second_same_label_idx])

                # Create a pair of different label
                different_cls_idx = (cls + random.randrange(1, 10)) % 10
                different_label_idx = classes_indices[different_cls_idx][cls_idx]
                x0_data.append(data[same_label_idx])
                x1_data.append(data[different_label_idx])
                new_targets.append(self.cls_to_idx['not_same'])
                x0_original_targets.append(self.targets[same_label_idx])
                x1_original_targets.append(self.targets[different_label_idx])

                pbar.update(1)

        pbar.close()
        x0_data = torch.stack(x0_data)
        x0_data = x0_data.reshape([-1, 1, 28, 28])
        x1_data = torch.stack(x1_data)
        x1_data = x1_data.reshape([-1, 1, 28, 28])
        new_targets = torch.from_numpy(np.array(new_targets))

        self.original_targets = x0_original_targets, x1_original_targets
        self.targets = new_targets
        self.data = x0_data, x1_data

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        original_target1, original_target2 = self.original_targets[0][index], self.original_targets[1][index]
        img1, img2, target = self.data[0][index], self.data[1][index], int(self.targets[index])

        img1 = Image.fromarray(img1.numpy().squeeze(), mode='L')
        img2 = Image.fromarray(img2.numpy().squeeze(), mode='L')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, target, original_target1, original_target2
