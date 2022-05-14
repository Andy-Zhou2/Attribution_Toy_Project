import torch as t
import torch
import torch.nn as nn
import math

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3 * a ** 3 - b ** 2
print(Q)
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)
print(a.grad)
print(t.randint(low=0, high=11, size=(1,)))
c = t.tensor((2, 2), dtype=t.int)
print(c.random_())
print(c)

d = t.tensor([1, 2, 3, 4])
print(d.size())
print(d.shape)
e = t.Tensor((4, 4))
print(e.size())
print(e.shape)

f = t.tensor(2)
print(f)
print(f.size())
print(f.shape)

print(t.randint(low=0, high=11, size=(5, 5)))



# Example of target with class indices
loss = nn.CrossEntropyLoss(reduction='none')
input = torch.rand(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
for i in range(3):
    s = 0
    s2 = 0
    for j in range(5):
        s += math.e**input[i][j]
    for j in range(5):
        s2 += math.e**input[i][j] / s
    print(math.e**input[i][target[i]] / s, s2)
output = loss(input, target)
# output.backward()
print(input, target, output)

a = t.tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
        [1.7401, -0.8805, -0.3402, -1.1936],
        [ 0.4907, -1.3948, -1.0691, -0.3132],
        [-1.6092,  0.5419, -0.2993,  0.3195]])
print(t.argmax(a))

m = nn.Sigmoid()
loss = nn.BCELoss()
input = t.randn(1024, 3, requires_grad=True)
target = t.empty(1024, 3).random_(2)
output = loss(m(input), target)
output.backward()
print(input, target, output)
print(input.grad)