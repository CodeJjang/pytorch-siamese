import torch
from torchvision.datasets import MNIST
from PIL import Image
import numpy as np
from tqdm import tqdm
import random


class TripletsMNIST(MNIST):
    def __init__(self, **kwargs):
        super(TripletsMNIST, self).__init__(**kwargs)

        self._gen_unique_labels()
        self._gen_label_to_indices()
        self._split_mnist_to_triplets()

    def _gen_unique_labels(self):
        self.unique_labels = np.unique(self.targets.numpy())

    def _gen_label_to_indices(self):
        self._label_to_indices_map = {}
        for label in self.unique_labels:
            self._label_to_indices_map[label] = np.where(self.targets.numpy() == label)[0]

    def _split_mnist_to_triplets(self):
        train_triplets = []
        train_labels = []
        for i in range(len(self.data)):
            (pos_anchor_img, pos_img, neg_img), ([anchor_label, pos_label, neg_label]) = self._generate_valid_triplet()
            train_triplets.append([pos_anchor_img, pos_img, neg_img])
            train_labels.append([anchor_label, pos_label, neg_label])

        self.data = train_triplets
        self.targets = train_labels

    def _generate_valid_triplet(self):
        positive_label = self.unique_labels[random.randint(0, len(self.unique_labels) - 1)]
        negative_label = positive_label
        while negative_label is positive_label:
            negative_label = self.unique_labels[random.randint(0, len(self.unique_labels) - 1)]

        positive_label_indices = self._label_to_indices_map[positive_label]
        anchor_index = positive_label_indices[random.randint(0, len(positive_label_indices) - 1)]
        positive_index = anchor_index
        while positive_index is anchor_index:
            positive_index = positive_label_indices[random.randint(0, len(positive_label_indices) - 1)]

        negative_label_indices = self._label_to_indices_map[negative_label]
        negative_index = negative_label_indices[random.randint(0, len(negative_label_indices) - 1)]

        anchor = self.data[anchor_index].numpy()
        positive = self.data[positive_index].numpy()
        negative = self.data[negative_index].numpy()

        return (anchor, positive, negative), (positive_label, positive_label, negative_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        anchor, positive, negative = self.data[index]
        anchor = np.expand_dims(anchor, axis=2)
        positive = np.expand_dims(positive, axis=2)
        negative = np.expand_dims(negative, axis=2)

        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return (anchor, positive, negative), self.targets[index]
