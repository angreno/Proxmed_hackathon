import torch

# create two 3D tensors
t1 = torch.randn(1.5,1.5,2)
t2 = torch.randn(1.5,1.5,2)

# dispaly the tensors
print(t1)
print("-------------------------------------------")
print(t2)
print("-------------------------------------------")
# concatenate the above tensors
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