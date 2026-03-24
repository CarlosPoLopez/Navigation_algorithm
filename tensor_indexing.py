import torch 

batch = 64
x = torch.rand(batch, 2, 5)
z = x.view(-1)
print(z)

