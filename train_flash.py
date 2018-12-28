import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import argparse

from scsg_step import *
from ncd_step import *
from model import *


# Training settings
parser = argparse.ArgumentParser(description='FLASH Example')

parser.add_argument('--BATCH-SIZE', type=int, default=100, metavar='N',
                    help='mini batch size for scsg in training (default: 100)')
parser.add_argument('--BATCH-SIZE-POWER', type=int, default=10, metavar='N',
                    help='batch size for power method in training (default: 10)')
parser.add_argument('--LARGE-BATCH-NUMBER', type=int, default=25, metavar='N',
                    help='how many mini-batch for calculate large batch gradient (default: 25)')
parser.add_argument('--LR-SCSG', type=float, default=0.6, metavar='LR',
                    help='learning rate for scsg (default: 0.6)')
parser.add_argument('--LR-PCA', type=float, default=0.5, metavar='LR',
                    help='learning rate for pca (default: 0.5)')
parser.add_argument('--LR-NEG', type=float, default=0.05, metavar='L',
                    help='learning rate for negative curvature descent (default: 0.05)')
parser.add_argument('--NORM-THRESHOLD', type=float, default=0.002, metavar='LR',
                    help='threshold for gradient norm (default: 0.002)')
parser.add_argument('--EPOCH', type=int, default=500, metavar='LR',
                    help='total epoch (data pass) for the algorithm (default: 500)')
parser.add_argument('--PCA-ITER', type=int, default=100, metavar='N',
                    help='iteration for PCA (default: 100)')
parser.add_argument('--LAMBDA-PCA', type=float, default=5.0, metavar='L',
                    help='normalization term for power method (default: 5.0)')
args = parser.parse_args()

# MNIST dataset
train_data = torchvision.datasets.MNIST(
    root='./data/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

test_data = torchvision.datasets.MNIST(
    root='./data/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

# Setup DataLoader
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=args.BATCH_SIZE,
                               shuffle=True)

train_loader_power = Data.DataLoader(dataset=train_data,
                                     batch_size=args.BATCH_SIZE_POWER,
                                     shuffle=True)

test_loader = Data.DataLoader(dataset=test_data,
                              batch_size=args.BATCH_SIZE,
                              shuffle=True)


def main():
    # define AutoEncoder net
    net = AutoEncoder().cuda()
    # define loss function
    loss_func = nn.MSELoss().cuda()
    # setup optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=args.LR_SCSG)

    # training
    for epoch in range(args.EPOCH):
        inner_iter_num = np.random.geometric(1.0/(args.LARGE_BATCH_NUMBER + 1.0))

        # take one epoch scsg step
        cur_train_loss, cur_test_loss, cur_grad_norm = scsg_step(net, optimizer, train_loader,
                                                                 test_loader, loss_func,
                                                                 inner_iter_num, args)

        # take negative curvature step if the gradient norm is small
        if cur_grad_norm < args.NORM_THRESHOLD:
            ncd3_step(net, optimizer, train_loader_power, loss_func, args)

        # print progress
        print('Epoch: ', epoch,
              '| train loss: %.8f' % cur_train_loss,
              '| test loss: %.8f' % cur_test_loss,
              '| gradient norm: %.8f' % cur_grad_norm)


if __name__ == '__main__':
    main()
