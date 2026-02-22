import torch

my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32, requires_grad=True)
#print(my_tensor)
 
x = torch.empty(size=(3,3))

x = torch.zeros((3,3))

x = torch.randint(0, 10, (3,3))

x = torch.ones((3,3))

x = torch.eye(3,3)

x = torch.arange(start=0, end=3, step=1)

x = torch.empty((1,5)).normal_(mean = 0, std = 1)

x = torch.diag((torch.randint(0, 5, (3,3))))
print(x)



