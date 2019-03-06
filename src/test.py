import torch
from torch.autograd import grad, Variable

a = torch.FloatTensor([1, 2, 3])
b = torch.FloatTensor([3, 4, 5])

a, b = Variable(a, requires_grad=True), Variable(b, requires_grad=True)

c = a**2 + b**2
c = c.sum()

grad_b = torch.autograd.grad(c, b, retain_graph=True, create_graph=True)
print(grad_b[0][0])

grad2_b = torch.autograd.grad(grad_b[0], b, retain_graph=True, create_graph=True)
print(grad2_b)
