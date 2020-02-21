import torch
from time import time

a = torch.ones(1000)
b = torch.ones(1000)

start = time()
c = torch.zeros(1000)

for i in range(1000):
    c[i] = a[i] + b[i]

start2 = time()
print(start2 - start)

d = a+b
print(float(time() - start2))