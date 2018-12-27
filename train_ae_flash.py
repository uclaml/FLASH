import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np
import copy
import torch.cuda.comm
import torch.tensor
import simplejson

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
BATCH_SIZE = 100
BATCH_SIZE_POWER = 10

# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./data/',
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=True,  # download it if you don't have it
)

test_data = torchvision.datasets.MNIST(
    root='./data/',
    train=False,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=True,  # download it if you don't have it
)


# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
train_loader_power = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE_POWER, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE_POWER, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # self.encoder = nn.Sequential(
        #     nn.Linear(28 * 28, 1024),
        #     nn.Softplus(20),
        #     nn.Linear(1024, 512),
        #     nn.Softplus(20),
        #     nn.Linear(512, 256),
        #     nn.Softplus(20),
        #     nn.Linear(256, 128),
        #     nn.Softplus(20),
        #     nn.Linear(128, 64),
        #     nn.Softplus(20),
        #     nn.Linear(64, 32),
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(32, 64),
        #     nn.Softplus(20),
        #     nn.Linear(64, 128),
        #     nn.Softplus(20),
        #     nn.Linear(128, 256),
        #     nn.Softplus(20),
        #     nn.Linear(256, 512),
        #     nn.Softplus(20),
        #     nn.Linear(512, 1024),
        #     nn.Softplus(20),
        #     nn.Linear(1024, 28 * 28),
        # )

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.Softplus(),
            nn.Linear(1024, 512),
            nn.Softplus(),
            nn.Linear(512, 256),
            nn.Softplus(),
            nn.Linear(256, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.Softplus(),
            nn.Linear(256, 512),
            nn.Softplus(),
            nn.Linear(512, 1024),
            nn.Softplus(),
            nn.Linear(1024, 28 * 28),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def partial_grad(self, input_data, loss_function):
        """
        Function to compute the stochastic gradient
        args : data, target, loss_function
        data and target should be shaped by Variable()
        return loss
        """
        input_data = input_data.view(-1, 28 * 28)
        output_data = self.forward(input_data)
        # compute the partial loss
        loss = loss_function(output_data, input_data)

        # compute gradient
        loss.backward()

        return loss

    def calculate_loss_grad(self, dataset, loss_function, large_batch_num):
        """
        Function to compute the large-batch loss and the large-batch gradient
        args : dataset, loss function and number of samples
        return : total loss and full grad norm
        """

        total_loss = 0.0
        full_grad_norm = 0.0

        num_batch = large_batch_num

        for data_i, data in enumerate(dataset):
            # only calculate the sub-sampled large batch
            if data_i > num_batch - 1:
                break

            inputs, labels = data
            # wrap data and target into variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            total_loss += (1.0 / num_batch) * self.partial_grad(inputs, loss_function).data[0]

        # calculate the norm of the large-batch gradient
        for param in self.parameters():
            full_grad_norm += param.grad.data.norm(2) ** 2

        full_grad_norm = np.sqrt(full_grad_norm) / num_batch

        # print('total loss:', total_loss)
        # print('full_grad_norm:', full_grad_norm)

        return total_loss, full_grad_norm


def power_method(net, dataset_power, loss_function, n_epoch, lr_pca, lambda_power):
    """
    Function to calculate the smallest eigenvalue of Hessian matrix H
    args : dataset, loss function, number of epochs, learning rate
    return : total_loss_epoch, grad_norm_epoch
    """

    # construct the iter point y_t
    iter_net = copy.deepcopy(net).cuda()

    # random init
    norm_init = 0.0
    for p_init in iter_net.parameters():
        p_init.data = torch.randn(p_init.data.size())
        norm_init += p_init.data.norm(2) ** 2
    norm_init = np.sqrt(norm_init)

    for p_iter in iter_net.parameters():
        p_iter.data /= norm_init

    iter_net.cuda()

    estimate_value = 0.0
    # scsg for PCA

    optimizer_PCA = torch.optim.SGD(iter_net.parameters(), lr=lr_pca, momentum=0.0)

    estimate_value_avg = 0.0
    estimate_epoch = 1.0
    input_data = 0.0

    for data_iter, data in enumerate(dataset_power):

        if data_iter > n_epoch - 1:
            break

        # get the input and label
        inputs, labels = data

        # wrap data and target into variable
        input_data, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        # zero the gradient
        net.zero_grad()
        input_data = input_data.view(-1, 28 * 28)
        outputs = net.forward(input_data)

        loss_self_defined = loss_function(outputs, input_data)

        # compute the gradient
        grad_hv = torch.autograd.grad(loss_self_defined, net.parameters(), create_graph=True)

        # compute the Hessian-vector product for current point
        inner_product = 0.0
        for p_iter_net, p_grad in zip(iter_net.parameters(), grad_hv):
            inner_product += torch.sum(p_iter_net * p_grad)

        h_v = torch.autograd.grad(inner_product, net.parameters(), create_graph=True)

        optimizer_PCA.zero_grad()
        outputs = iter_net.forward(input_data)
        current_loss = loss_function(outputs, input_data)
        current_loss.backward()

        estimate_value = 0.0
        # estimate the curvature
        for p_iter, p_hv in zip(iter_net.parameters(), h_v):
            estimate_value += torch.sum(p_iter * p_hv)
            p_iter.grad.data = - p_iter.data + p_hv.data / lambda_pca
        estimate_value = float(estimate_value)

        optimizer_PCA.step()

        epoch_len = n_epoch / 10.0
        # print every epoch_len mini-batches
        if data_iter % epoch_len == epoch_len - 8:
            print('epoch: %d, estimate_value: %.8f' % (epoch, estimate_value_avg/estimate_epoch))
            estimate_value_avg = 0.0
            estimate_epoch = 1.0
        estimate_value_avg += estimate_value
        estimate_epoch += 1.0

        norm_iter = 0.0
        for p_iter in iter_net.parameters():
            norm_iter += p_iter.data.norm(2) ** 2
        norm_iter = np.sqrt(norm_iter)

        # normalization for iter_net
        for p_iter in iter_net.parameters():
            p_iter.data /= (1.0 * norm_iter)

    estimate_value = estimate_value_avg / estimate_epoch
    num_data_pass = n_epoch / 10.0

    # calculate full gradient
    net.zero_grad()
    net.calculate_loss_grad(dataset_power, loss_function, num_data_pass)

    # update with negative curvature
    direction_value = 0.0

    if estimate_value < 0.0:
        for p_net, p_iter in zip(net.parameters(), iter_net.parameters()):
            direction_value += torch.dot(p_net.grad, p_iter)
        print('direction_value:', float(direction_value))
        direction_value = float(torch.sign(direction_value))

        net.zero_grad()
        outputs = net.forward(input_data)
        current_loss = loss_function(outputs, input_data)
        current_loss.backward()
        print('direction_value:', direction_value)
        for p_net, p_iter in zip(net.parameters(), iter_net.parameters()):
            p_net.grad.data = direction_value * p_iter.data * 0.1
        return estimate_value


def scsg_step(net, optimizer, dataset, test_dataset, loss_function, learning_rate, large_batch_num, inner_iter_num):
        """
        Function to updated weights with a SCSG backpropagation
        args : dataset, loss function, number of epochs, learning rate
        return : total_loss_epoch, grad_norm_epoch
        """
        # record previous_net_grad
        pre_net_full = copy.deepcopy(net)
        # record previous_net_sgd
        pre_net_mini = copy.deepcopy(net)

        # Compute full grad
        # set the gradient as 0
        pre_net_full.zero_grad()

        # take backward by using calculate_loss_grad()
        total_loss_epoch, grad_norm_epoch = pre_net_full.calculate_loss_grad(dataset, loss_function, large_batch_num)


        running_loss = 0.0
        epoch_num = 0.0
        # Run over the dataset
        for batch_id, batch_data in enumerate(dataset):

            if batch_id > inner_iter_num - 1:
                break

            # get the input and label
            inputs, labels = batch_data

            # wrap data and target into variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # compute previous stochastic gradient
            pre_net_mini.zero_grad()
            # take backward
            pre_net_mini.partial_grad(inputs, loss_function)

            # compute current stochastic gradient
            optimizer.zero_grad()
            inputs = inputs.view(-1, 28 * 28)
            outputs = net.forward(inputs)
            current_loss = loss_function(outputs, inputs)
            current_loss.backward()

            # take SCSG gradient step
            for p_net, p_mini, p_full in zip(net.parameters(), pre_net_mini.parameters(), pre_net_full.parameters()):
                p_net.grad.data += p_full.grad.data * (1.0 / large_batch_num) - p_mini.grad.data
            optimizer.step()

            # print statistics
            running_loss += current_loss.data[0]
            epoch_num += 1.0
            epoch_len = large_batch_num / 2.0
            # if batch_id % epoch_len == epoch_len - 2:  # print every epoch_len mini-batches
                # print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_id + 1, running_loss/epoch_num))

        net.zero_grad()
        test_loss_epoch, _ = net.calculate_loss_grad(test_dataset, loss_function, 100)
        num_full_batch = len(dataset)

        return running_loss/epoch_num, grad_norm_epoch, test_loss_epoch


autoencoder = AutoEncoder()
loss_func = nn.MSELoss().cuda()
autoencoder.cuda()

# SCSG - Parameters
lr_scsg = 0.6
large_batch_number = 25
inner_iter_num = large_batch_number * 9

# PowerMethod - Parameters
epoch_pca = 1000
lr_pca = 0.5
lambda_pca = 0.1
power_num = 0
count_neg = 0

# create loss variable
losses = []
test_losses = []
EPOCH = 50

optimizer = torch.optim.SGD(autoencoder.parameters(), lr=lr_scsg)


for epoch in range(EPOCH):
    cur_loss, cur_grad_norm, cur_test_loss = scsg_step(autoencoder, optimizer, train_loader, test_loader, loss_func,
                                                       lr_scsg, large_batch_number, inner_iter_num)
    losses.append(cur_loss.cpu().item())
    test_losses.append(cur_test_loss.cpu().item())

    print('Epoch: ', epoch, '| train loss: %.8f' % cur_loss, '| test loss: %.8f' % cur_test_loss)

    if cur_grad_norm < 0.005 and power_num % 20 == 0:
        power_num += 1
        count_neg += 1
        power_method(autoencoder, train_loader_power, loss_func, epoch_pca, lr_pca, lambda_pca)
        optimizer.step()

    if cur_grad_norm < 0.005:
        power_num += 1


f_train = open('small_train_neon_1.txt', 'w')
simplejson.dump(losses, f_train)
f_train.close()

f_test = open('small_test_neon_1.txt', 'w')
simplejson.dump(test_losses, f_test)
f_test.close()

