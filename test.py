import torch as t
from dataset import get_random_matrix_trace, fit_matrix_linear
from model import TraceNet
import numpy as np

net = TraceNet(10)

# matrix = t.tensor([[1, 2], [3, 4]], dtype=t.float)
# # matrix *= 10
# correct_trace = matrix[0][0] + matrix[1][1]
matrix, correct_trace = get_random_matrix_trace(10)
correct_trace = correct_trace.item()
print('generated matrix & trace:', matrix, correct_trace)

fitted_matrix = fit_matrix_linear(matrix, len(matrix), 10)
fitted_matrix = t.unsqueeze(fitted_matrix, 0)
fitted_matrix.requires_grad = True

trace_abs_diff = []
trace_diff_squares = []
diagonal_grad_diff_squares = []
non_diagonal_gard_diff_squares = []

for net_num in range(5, 301, 5):
    print('net', net_num)
    net.load_state_dict(t.load(f'./nets/TraceNet_{net_num}.net')['weight'])

    pred_trace = net(fitted_matrix)
    print(pred_trace.item())

    fitted_matrix.grad = None
    pred_trace.backward()
    g = fitted_matrix.grad.numpy()
    g = g[0]  # batch size is 1, take the 1st data
    # print(g)

    # collecting data
    trace_diff_squares.append((pred_trace - correct_trace) ** 2)
    trace_abs_diff.append(abs(pred_trace - correct_trace))

    diag = 0
    non_diag = 0

    for i in range(10):
        for j in range(10):
            if i == j:
                diag += (1 - g[i * 10 + j]) ** 2
            else:
                non_diag += g[i * 10 + j] ** 2

    diagonal_grad_diff_squares.append(diag)
    non_diagonal_gard_diff_squares.append(non_diag)

from matplotlib import pyplot as plt

net_nums = [i for i in range(5, 301, 5)]

plt.title('trace abs difference')
plt.plot(net_nums, trace_abs_diff)
plt.savefig('./data/trace abs difference.png')
plt.show()
plt.close()

plt.title('trace square difference')
plt.plot(net_nums, trace_diff_squares)
plt.savefig('./data/trace square difference.png')
plt.show()
plt.close()

plt.title('diagonal square difference')
plt.plot(net_nums, diagonal_grad_diff_squares)
plt.savefig('./data/diagonal square difference.png')
plt.show()
plt.close()

plt.title('non-diagonal square difference')
plt.plot(net_nums, non_diagonal_gard_diff_squares)
plt.savefig('./data/non-diagonal square difference.png')
plt.show()
plt.close()

g = g.reshape([10, 10])
print(g)
plt.imshow(g)
plt.colorbar()
plt.savefig('./data/image representation of gradient.png')
plt.show()
plt.close()