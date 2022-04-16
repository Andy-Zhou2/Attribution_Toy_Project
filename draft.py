import torch as t
import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2
print(Q)
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)
print(a.grad)
print(t.randint(low=0, high=11, size=(1,)))
c = t.tensor((2, 2), dtype=t.int)
print(c.random_())
print(c)

d = t.tensor([1,2,3,4])
print(d.size())
print(d.shape)
e = t.Tensor((4,4))
print(e.size())
print(e.shape)

f = t.tensor(2)
print(f)
print(f.size())
print(f.shape)