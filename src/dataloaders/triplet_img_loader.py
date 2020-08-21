import os
import numpy as np

import torch.utils.data
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST

from src.datasets.MNIST import MNIST as MY_MNIST


class TripletMNISTLoader(torch.utils.data.Dataset):
    def __init__(self, triplets, transform=None, labels=None):
        self.triplets = triplets
        self.transform = transform
        self.labels = labels

    def __getitem__(self, index):
        img1, img2, img3 = self.triplets[index]
        img1 = np.expand_dims(img1, axis=2)
        img2 = np.expand_dims(img2, axis=2)
        img3 = np.expand_dims(img3, axis=2)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        if self.labels is None:
            return img1, img2, img3
        else:
            return (img1, img2, img3), self.labels[index]

    def __len__(self):
        return len(self.triplets)


def get_loader(args, use_cuda, data_path):
    train_data_loader = None
    test_data_loader = None

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_triplets = []
    test_triplets = []
    train_labels = []
    test_labels = []

    dset_obj = None
    # means = (0.485, 0.456, 0.406)
    # stds = (0.229, 0.224, 0.225)
    means = (0.485,)
    stds = (0.229,)

    train_dataset = MNIST(data_path, train=True, download=True)
    # test_dataset = MNIST(data_path, train=False, download=True)
    test_dataset = MNIST(data_path, train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(means, stds)
                         ])
                         )
    dset_obj = MY_MNIST(train_dataset, test_dataset)
    loader = TripletMNISTLoader
    means = (0.485,)
    stds = (0.229,)

    dset_obj.load()
    for i in range(len(train_dataset)):
        (pos_anchor_img, pos_img, neg_img), ([anchor_label, pos_label, neg_label]) = dset_obj.getTriplet()
        train_triplets.append([pos_anchor_img, pos_img, neg_img])
        train_labels.append([anchor_label, pos_label, neg_label])
    for i in range(len(test_dataset)):
        (pos_anchor_img, pos_img, neg_img), ([anchor_label, pos_label, neg_label]) = dset_obj.getTriplet(split='test')
        test_triplets.append([pos_anchor_img, pos_img, neg_img])
        test_labels.append([anchor_label, pos_label, neg_label])
    train_data_loader = torch.utils.data.DataLoader(
        loader(train_triplets,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize(means, stds)
               ]),
               labels=train_labels),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_data_loader = torch.utils.data.DataLoader(
    #     loader(test_triplets,
    #            transform=transforms.Compose([
    #                transforms.ToTensor(),
    #                transforms.Normalize(means, stds)
    #            ]),
    #            labels=test_labels),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    return train_data_loader, test_data_loader
