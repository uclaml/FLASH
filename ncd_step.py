import torch
from torch.autograd import Variable
import copy
import numpy as np


def ncd3_step(net, optimizer, train_loader_power, loss_function, args):
    """
    Function to take the negative curvature step
    args : net, train loader, loss function, number of iterations, learning rate
           lambda (parameter for pca), lr_scsg, lr_ncd3
    return : estimate_value, i.e., v^{\top}*H*v
    """
    # print('Calculating Negative Curvature')

    n_iter = args.POWER_ITER
    lr_pca = args.LR_PCA
    lambda_pca = args.LAMBDA_PCA
    lr_scsg = args.LR_SCSG
    lr_ncd3 = args.LR_PCA

    # construct the iterative net
    iter_net = copy.deepcopy(net).cuda()

    # random init with \|w\|_{2} = 1
    norm_init = 0.0
    for p_init in iter_net.parameters():
        p_init.data = torch.randn(p_init.data.size())
        norm_init += p_init.data.norm(2) ** 2
    norm_init = np.sqrt(norm_init)
    for p_iter in iter_net.parameters():
        p_iter.data /= norm_init
    iter_net.cuda()

    # optimizer for PCA
    optimizer_PCA = torch.optim.SGD(iter_net.parameters(), lr=lr_pca, momentum=0.0)

    estimate_value_avg = 0.0
    estimate_iter = 1.0
    input_data = 0.0

    for data_idx, data in enumerate(train_loader_power):

        if data_idx > n_iter - 1:
            break

        # load the input
        inputs, _ = data
        input_data = Variable(inputs).cuda()

        # compute the Hessian-vector product for current point
        net.zero_grad()
        input_data = input_data.view(-1, 28 * 28)
        outputs = net(input_data)
        loss_self_defined = loss_function(outputs, input_data)
        grad_hv = torch.autograd.grad(loss_self_defined, net.parameters(), create_graph=True)
        inner_product = 0.0
        for p_iter_net, p_grad in zip(iter_net.parameters(), grad_hv):
            inner_product += torch.sum(p_iter_net * p_grad)
        h_v = torch.autograd.grad(inner_product, net.parameters(), create_graph=True)

        # take a gradient ascent step
        optimizer_PCA.zero_grad()
        outputs = iter_net.forward(input_data)
        current_loss = loss_function(outputs, input_data)
        current_loss.backward()
        estimate_value = 0.0
        for p_iter, p_hv in zip(iter_net.parameters(), h_v):
            estimate_value += torch.sum(p_iter * p_hv)
            p_iter.grad.data = - p_iter.data + p_hv.data / lambda_pca
        estimate_value = float(estimate_value)
        optimizer_PCA.step()

        # update estimate value
        epoch_len = n_iter / 10.0
        # print every epoch_len mini-batches
        if data_idx % epoch_len == 0:
            # print('epoch: %d, estimate_value: %.8f' % (epoch, estimate_value_avg/estimate_iter))
            estimate_value_avg = 0.0
            estimate_iter = 1.0
        estimate_value_avg += estimate_value
        estimate_iter += 1.0

        # calculate norm
        norm_iter = 0.0
        for p_iter in iter_net.parameters():
            norm_iter += p_iter.data.norm(2) ** 2
        norm_iter = torch.sqrt(norm_iter)

        # normalization for iter_net
        for p_iter in iter_net.parameters():
            p_iter.data /= (1.0 * norm_iter)

    estimate_value = estimate_value_avg / estimate_iter

    # calculate full gradient
    net.zero_grad()
    net.calculate_loss_grad(train_loader_power, loss_function, n_iter)

    # update with negative curvature
    direction_value = 0.0

    for p_net, p_iter in zip(net.parameters(), iter_net.parameters()):
        direction_value += torch.sum(p_net.grad * p_iter)
    direction_value = float(torch.sign(direction_value))

    net.zero_grad()
    outputs = net.forward(input_data)
    current_loss = loss_function(outputs, input_data)
    current_loss.backward()
    for p_net, p_iter in zip(net.parameters(), iter_net.parameters()):
        p_net.grad.data = direction_value * p_iter.data * (lr_ncd3 / lr_scsg)

    optimizer.step()
    return estimate_value

