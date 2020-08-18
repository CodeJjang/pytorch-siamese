import torch
from torchvision.datasets import MNIST
from PIL import Image
import numpy as np
from tqdm import tqdm
import random


class TripletsMNIST(MNIST):
    def __init__(self, **kwargs):
        super(TripletsMNIST, self).__init__(**kwargs)
        self._shuffle()
        self._split_mnist_to_triplets()

    def _split_mnist_to_triplets(self):
        data = self.data
        classes_indices = [np.where(self.targets == i)[0] for i in range(10)]
        orig_classes_len = len(self.classes)

        new_data = []
        original_targets = []

        self.classes = []
        pairs_per_class_amount = min([len(classes_indices[d]) for d in range(10)]) - 1

        # Init pbar data
        pbar = tqdm(total=int(orig_classes_len * pairs_per_class_amount), desc='Creating MNIST triplet samples')

        for cls_idx in range(pairs_per_class_amount):
            for cls in range(orig_classes_len):
                anchor_label_idx = classes_indices[cls][cls_idx]
                positive_label_idx = classes_indices[cls][cls_idx + 1]
                negative_cls = (cls + random.randrange(1, 10)) % 10
                negative_label_idx = classes_indices[negative_cls][cls_idx]

                # Create a triplet of a,p,n
                anchor = data[anchor_label_idx]
                positive = data[positive_label_idx]
                negative = data[negative_label_idx]

                new_data.append(torch.stack([anchor, positive, negative]))
                original_targets.append(torch.stack([self.targets[anchor_label_idx],
                                                     self.targets[positive_label_idx],
                                                     self.targets[negative_label_idx]]))

                pbar.update(1)

        pbar.close()
        new_data = torch.stack(new_data)
        original_targets = torch.stack(original_targets)

        self.targets = original_targets
        self.data = new_data

    def _shuffle(self):
        perm = np.random.permutation(len(self.data))
        self.data = self.data[perm]
        self.targets = self.targets[perm]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        anchor, positive, negative = self.data[index]

        anchor = Image.fromarray(anchor.numpy().squeeze(), mode='L')
        positive = Image.fromarray(positive.numpy().squeeze(), mode='L')
        negative = Image.fromarray(negative.numpy().squeeze(), mode='L')

        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        processed_data = torch.stack([anchor, positive, negative])
        return processed_data, self.targets[index]
