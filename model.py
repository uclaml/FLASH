import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.Softplus(10),
            nn.Linear(1024, 512),
            nn.Softplus(10),
            nn.Linear(512, 256),
            nn.Softplus(10),
            nn.Linear(256, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.Softplus(10),
            nn.Linear(256, 512),
            nn.Softplus(10),
            nn.Linear(512, 1024),
            nn.Softplus(10),
            nn.Linear(1024, 28 * 28),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def partial_grad(self, input, loss_function):
        """
        Function to compute the stochastic gradient
        args : input, loss_function
        return : loss
        """
        input = input.view(-1, 28 * 28)
        output = self.forward(input)
        # compute the partial loss
        loss = loss_function(output, input)

        # compute gradient
        loss.backward()

        return loss

    def calculate_loss_grad(self, dataset, loss_function, large_batch_num):
        """
        Function to compute the large-batch loss and the large-batch gradient
        args : dataset, loss function, number of samples to be calculated
        return : total loss and full grad norm
        """

        total_loss = 0.0
        full_grad_norm = 0.0

        for idx, data in enumerate(dataset):
            # only calculate the sub-sampled large batch
            if idx > large_batch_num - 1:
                break
            # load input
            inputs, _ = data
            inputs = Variable(inputs).cuda()
            # calculate loss
            total_loss += self.partial_grad(inputs, loss_function).item()

        total_loss /= large_batch_num

        # calculate the norm of the large batch gradient
        for param in self.parameters():
            full_grad_norm += param.grad.data.norm(2) ** 2

        full_grad_norm = np.sqrt(full_grad_norm).item() / large_batch_num

        return total_loss, full_grad_norm
