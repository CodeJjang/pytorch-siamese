from __future__ import print_function
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from src.datasets.TripletsMNIST import TripletsMNIST
from src.models.SiameseNetwork import SiameseNetwork
from src.models.knn import KNN
from src.utils.Files import create_dir_path_if_not_exist
from src.sampling.BatchHard import BatchHard


def train(args, model, device, train_loader, optimizer, criterion, epochs, test_loader, knn, sampling):
    curr_time = str(datetime.datetime.now()).replace(' ', '_')
    best_model = None
    best_acc = None
    for epoch in range(1, epochs + 1):
        model.train()
        train_embeddings = []
        train_original_labels = []
        for batch_idx, (data, original_targets) in enumerate(train_loader):
            data = [_data.to(device) for _data in data]
            optimizer.zero_grad()
            output = model(data[0], data[1], data[2])

            # Collect embeddings and original labels for KNN
            train_embeddings += [out.cpu().detach().numpy().copy() for out in output]
            train_original_labels += [target for target in original_targets]

            if epoch > 1:
                output = sampling(output, original_targets)
            loss = criterion(output[0], output[1], output[2])
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                if args.dry_run:
                    break

        train_embeddings = np.concatenate(train_embeddings)
        train_original_labels = np.concatenate(train_original_labels)
        test_embeddings, outputs, test_acc = test(model, knn, device, test_loader, train_embeddings,
                                                  train_original_labels)

        # Plot train and test clusters for debugging
        fname = f'{curr_time}_{epoch}'
        plot_mnist(args.plot_path, f'{fname}_train', train_embeddings, train_original_labels)
        plot_mnist(args.plot_path, f'{fname}_test', test_embeddings, outputs)


def test(model, knn, device, test_loader, train_embeddings, train_labels):
    model.eval()
    test_embeddings = []
    test_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            test_embeddings += output.cpu().numpy().tolist()
            test_labels += target

    test_embeddings = np.array(test_embeddings)
    test_labels = np.array(test_labels)
    correct, acc = knn(train_embeddings, train_labels, test_embeddings, test_labels)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * acc))
    return test_embeddings, test_labels, acc


def get_embeddings(model, device, test_loader):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            embeddings += output.cpu().numpy().tolist()
            labels += target.cpu().numpy().tolist()
    return np.array(embeddings), np.array(labels)


def load_datasets(data_dir, train_transform, test_transform):
    train_set = TripletsMNIST(root=data_dir, train=True, download=True,
                              transform=train_transform)

    test_set = datasets.MNIST(root=data_dir, train=False, download=True,
                              transform=test_transform)

    return train_set, test_set


def plot_mnist(out_dir, fname, embeddings, labels):
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']

    for i in range(10):
        f = embeddings[np.where(labels == i)]
        plt.plot(f[:, 0], f[:, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.savefig(os.path.join(out_dir, f'{fname}.png'))
    plt.clf()


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST classifier using a Siamese network')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='MO',
                        help='momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model-path', default='saved_models/',
                        help='Models location')
    parser.add_argument('--plot-path', default='plots/',
                        help='Plot location')
    parser.add_argument('--data-path', default='data/',
                        help='Data location')
    parser.add_argument('--knn', type=int, default=3,
                        help='KNN neighbours (default: 3)')
    parser.add_argument('--triplet-margin', type=float, default=0.2,
                        help='Triplet loss margin (default: 0.2)')
    parser.add_argument('--semi-hard', action='store_true', default=False,
                        help='Whether to mine semi hard negative samples')
    parser.add_argument('--mine-all-anchor-positives', action='store_true', default=False,
                        help='Whether to mine all anchor positive pairs or just randomly pick them')
    return parser.parse_args()


def print_train_stats(args, device):
    print(f'Using device {device}')

    if args.semi_hard:
        print('Mining all semi-hard negative samples')
    else:
        print('Mining all hard negative samples')

    if args.mine_all_anchor_positives:
        print('Mining all possible anchor-positive pair combinations')
    else:
        print('Mining random anchor-positive pairs')


def main():
    args = parse_args()

    create_dir_path_if_not_exist(args.model_path)
    create_dir_path_if_not_exist(args.plot_path)
    create_dir_path_if_not_exist(args.data_path)

    model_path = os.path.join(args.model_path, 'pairwise_siamese.pt')

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print_train_stats(args, device)

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True
                       },
                      )

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set, test_set = load_datasets(args.data_path, train_transform, test_transform)
    train_loader = torch.utils.data.DataLoader(train_set, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, **kwargs)

    model = SiameseNetwork().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    criterion = nn.TripletMarginLoss(margin=args.triplet_margin)
    sampling = BatchHard(args.triplet_margin, args.semi_hard, args.mine_all_anchor_positives)
    knn = KNN(args.knn)
    train(args, model, device, train_loader, optimizer, criterion, args.epochs, test_loader, knn, sampling)

    if args.save_model:
        torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()
