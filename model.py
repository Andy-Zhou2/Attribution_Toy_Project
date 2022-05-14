import torch as t
import torch.nn as nn


class TraceNet(nn.Module):
    def __init__(self, matrix_size):
        """
        :param matrix_size: size of one side of square matrix (for 10x10 matrix it is 10)
        """
        super(TraceNet, self).__init__()
        self.matrix_size = matrix_size
        self.fc1 = nn.Linear(matrix_size * matrix_size, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 11)

    def forward(self, x):
        """
        :param x: input tensor of size (batch_size, matrix_size * matrix_size)
        :return: tensor of size (batch_size, 1)
        """
        x = t.sigmoid(self.fc1(x))
        x = t.sigmoid(self.fc2(x))
        x = t.sigmoid(self.fc3(x))
        x = self.fc4(x)

        return x
