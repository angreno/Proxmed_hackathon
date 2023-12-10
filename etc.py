import torch

t1 = torch.randn(1.5,1.5,2)
t2 = torch.randn(1.5,1.5,2)
print(t1)
print("-------------------------------------------")
print(t2)
print("-------------------------------------------")

t = torch.cat((t1,t2), 0)
print(t)
print(t.shape)
print("-------------------------------------------")
t = torch.cat((t1,t2), 1)
print(t)
print(t.shape)
print("-------------------------------------------")
t = torch.cat((t1,t2), 2)
print(t)
print(t.shape)
