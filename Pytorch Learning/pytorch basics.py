import torch

# Vector + element by element multiplication
x = torch.tensor([5, 3])
y = torch.tensor([2, 1])
print(x*y)

# Zeros + shape
x = torch.zeros([2, 5])
print(x)
print(x.shape)

# Random vector
y = torch.rand([2, 5])
print(y)

# Reshape
y = y.view([1, 10])
print(y)