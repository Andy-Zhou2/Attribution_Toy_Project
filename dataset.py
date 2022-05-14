import torch as t
from torch.utils.data import Dataset
from random import randint


def get_random_matrix_trace(size):
    a = t.randint(low=0, high=6, size=(size, size))
    trace = 0
    for i in range(size):
        trace += a[i, i]
    return a, trace


def fit_matrix_linear(matrix, matrix_size, target_step_size):
    r = t.zeros((target_step_size * target_step_size))
    for i in range(matrix_size):
        r[i * target_step_size:i * target_step_size + matrix_size] = matrix[i, :]
    return r


class RandomMatrixDataset(Dataset):
    def __init__(self, max_matrix_size):
        self.max_matrix_size = max_matrix_size

    def __len__(self):
        return 1024

    def __getitem__(self, index):
        # returns a random tensor
        size = 2  # always generate 2x2 matrices
        a, trace = get_random_matrix_trace(size)
        r = fit_matrix_linear(a, size, self.max_matrix_size)
        trace_vector = t.zeros(11)  # possible trace: from 0-20
        trace_vector[trace] = 1
        return r, trace_vector, trace

# md = RandomMatrixDataset(10)
# m, tt, t2 = md[0]
# print(m.reshape(10, 10), t2)
