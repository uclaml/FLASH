import copy
from torch.autograd import Variable


def scsg_step(net, optimizer, train_loader, test_loader, loss_function, inner_iter_num, args):
    """
    Function to updated weights with a SCSG backpropagation
    args : train_loader, test_loader, loss function, number of epochs,
    return : total_loss_epoch, grad_norm_epoch
    """
    # record previous net full gradient
    pre_net_full = copy.deepcopy(net)
    # record previous net mini batch gradient
    pre_net_mini = copy.deepcopy(net)

    large_batch_num = args.LARGE_BATCH_NUMBER

    # Compute full grad
    pre_net_full.zero_grad()
    _, grad_norm_lb = pre_net_full.calculate_loss_grad(train_loader, loss_function, large_batch_num)

    running_loss = 0.0
    iter_num = 0.0

    # Run over the train_loader
    for batch_id, batch_data in enumerate(train_loader):

        if batch_id > inner_iter_num - 1:
            break

        # get the input and label
        inputs, _ = batch_data

        # wrap data and target into variable
        inputs = Variable(inputs).cuda()

        # compute previous stochastic gradient
        pre_net_mini.zero_grad()
        # take backward
        pre_net_mini.partial_grad(inputs, loss_function)

        # compute current stochastic gradient
        optimizer.zero_grad()

        inputs = inputs.view(-1, 28 * 28)
        outputs = net(inputs)
        current_loss = loss_function(outputs, inputs)
        current_loss.backward()

        # take SCSG gradient step
        for p_net, p_mini, p_full in zip(net.parameters(), pre_net_mini.parameters(), pre_net_full.parameters()):
            p_net.grad.data += p_full.grad.data * (1.0 / large_batch_num) - p_mini.grad.data
        optimizer.step()

        # print statistics
        running_loss += current_loss.item()
        iter_num += 1.0

    # calculate training loss
    train_loss = running_loss / iter_num

    # calculate test loss
    net.zero_grad()
    test_loss, _ = net.calculate_loss_grad(test_loader, loss_function, len(test_loader)/args.BATCH_SIZE)

    return train_loss, test_loss, grad_norm_lb
