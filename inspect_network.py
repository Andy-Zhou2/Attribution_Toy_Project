import torch as t
from dataset import get_random_matrix_trace, fit_matrix_linear
from model import TraceNet
import numpy as np

net = TraceNet(2)
net.load_state_dict(t.load('./nets/TraceNetClassification_size2_0-5_BCE_1e-3_1000.net')['weight'])

matrix = t.tensor([[4, 2], [3, 2]], dtype=t.float)
matrix *= 1

fitted_matrix = fit_matrix_linear(matrix, 2, 2)
fitted_matrix = t.unsqueeze(fitted_matrix, 0)
fitted_matrix.requires_grad = True

print(fitted_matrix)
pred_trace = net(fitted_matrix)
print(pred_trace)
print(t.argmax(pred_trace))
pred_trace.backward()
g = fitted_matrix.grad.numpy()
print(g)

for i in range(10):
    print(i, g[0][i * 10 + i])

print(g[0].shape)
print(np.sort(g[0]))
