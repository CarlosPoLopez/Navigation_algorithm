import torch 

x = torch.tensor([3,6,1])
y = torch.tensor([[4,5,6],[4,5,6]])


x1 = torch.randint(0,10,(2,3))
x2 = torch.randint(0,10,(3,5))
#print((torch.mm(x1,x2)))

sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
sorted_y, indices = torch.sort(x, dim = 0, descending=False)
print(sorted_y, indices)

