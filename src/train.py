from __future__ import print_function
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from src.datasets.PairsMNIST import PairsMNIST
from src.losses.ContrastiveLoss import ContrastiveLoss
from src.models.SiameseNetwork import SiameseNetwork
from src.models.SiameseNetwork2 import SiameseNetwork2
from src.utils.Files import create_dir_path_if_not_exist


def train(args, model, device, train_loader, optimizer, criterion, epochs, test_loader, scheduler):
    curr_time = str(datetime.datetime.now()).replace(' ', '_')
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data1, data2, target) in enumerate(train_loader):
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data1, data2)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data1), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                if args.dry_run:
                    break
        # test(model, device, test_loader)
        embeddings, outputs = get_embeddings(model, device, test_loader)
        fname = f'{curr_time}_{epoch}'
        plot_mnist(args.plot_path, fname, embeddings, outputs)
        scheduler.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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


def load_datasets(data_dir, train_set_cache_path, test_set_cache_path, transform):
    try:
        train_set = torch.load(train_set_cache_path)
    except:
        train_set = PairsMNIST(root=data_dir, train=True, download=True,
                               transform=transform)
        torch.save(train_set, train_set_cache_path)

    try:
        test_set = torch.load(test_set_cache_path)
    except:
        test_set = datasets.MNIST(root=data_dir, train=False, download=True,
                                  transform=transform)
        torch.save(test_set, test_set_cache_path)

    return train_set, test_set


def plot_mnist(out_dir, fname, embeddings, labels):
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']

    for i in range(10):
        f = embeddings[np.where(labels == i)]
        plt.plot(f[:, 0], f[:, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.savefig(os.path.join(out_dir, f'{fname}.png'))


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST classifier using a Siamese network')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='MO',
                        help='momentum (default: 0.9)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.1)')
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
    parser.add_argument('--cache', default='cache/',
                        help='Cache location')
    parser.add_argument('--model-path', default='models/',
                        help='Models location')
    parser.add_argument('--plot-path', default='plots/',
                        help='Plot location')
    parser.add_argument('--data-path', default='data/',
                        help='Data location')
    return parser.parse_args()


def main():
    args = parse_args()

    create_dir_path_if_not_exist(args.cache)
    create_dir_path_if_not_exist(args.model_path)
    create_dir_path_if_not_exist(args.plot_path)
    create_dir_path_if_not_exist(args.data_path)

    train_set_cache_path = os.path.join(args.cache, 'train_set.p')
    test_set_cache_path = os.path.join(args.cache, 'test_set.p')
    model_path = os.path.join(args.model_path, 'pairwise_siamese.pt')

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                      )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set, test_set = load_datasets(args.data_path, train_set_cache_path, test_set_cache_path, transform)
    train_loader = torch.utils.data.DataLoader(train_set, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **kwargs)

    model = SiameseNetwork().to(device)
    # model = SiameseNetwork2().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    criterion = ContrastiveLoss()
    train(args, model, device, train_loader, optimizer, criterion, args.epochs, test_loader, scheduler)

    if args.save_model:
        torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()
