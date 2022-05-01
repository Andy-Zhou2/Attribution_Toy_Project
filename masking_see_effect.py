from dataset import get_random_matrix_trace
import pickle
import torch as t
from dataset import get_random_matrix_trace, fit_matrix_linear
from model import TraceNet
import numpy as np
from matplotlib import pyplot as plt

# code used to generate the matrix and store into a file

# matrix, trace = get_random_matrix_trace(10)
# print(matrix, trace)
#
# with open('masking_test_matrix', 'wb') as file:
#     pickle.dump({'matrix': matrix, 'trace': trace}, file)

# with open('masking_test_matrix', 'rb') as file:
#     d = pickle.load(file)

MUL_FACTOR = 1000
GPU = t.device("cuda:0")
matrix = t.ones([100]).to(GPU)
matrix *= MUL_FACTOR
correct_trace = 10
correct_trace *= MUL_FACTOR

net = TraceNet(10).to(GPU)
t.no_grad()  # no need for gradient calculation
abs_err = np.zeros([10, 10])
square_err = np.zeros([10, 10])

for i in range(10):
    for j in range(10):
        net.load_state_dict(t.load(f'./nets/TraceNet_mask_300_{i}_{j}.net')['weight'])
        pred_trace = net(matrix).item()
        print(i, j, pred_trace)
        abs_err[i, j] = abs(pred_trace - correct_trace)
        square_err[i, j] = (pred_trace - correct_trace) ** 2

plt.title(f'absolute error (truth={correct_trace})')
plt.imshow(abs_err)
plt.colorbar()
plt.savefig(f'./data/masking absolute error trace={correct_trace}.png')
plt.show()
plt.close()

plt.title(f'square error (truth={correct_trace})')
plt.imshow(square_err)
plt.colorbar()
plt.savefig(f'./data/masking square error trace={correct_trace}.png')
plt.show()
plt.close()
