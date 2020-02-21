import torch as th
import numpy as np

# a = th.tensor([5.5,6])
# a = a.new_ones(5,3,dtype = th.float64)


# y = th.eye(5,5)
# c = y
# k = th.gather(c,0,th.LongTensor([[1,1,0,1,2],[1,1,0,1,2]]))
# t = k.clone().view(-1,10)
# m = th.randn(1)  #讲一个数转换成 Python number
# m = np.ones([3,3])
# n = th.tensor(m)
# n += 1
# print(n,m)

# x = th.tensor([1,2,3])
# if  th.cuda.is_available():
#     device = th.device("cuda")
#     y = th.ones_like(x,device=device)
#     x = x.to(device)
#     z = x + y
#     print(z.numpy())
#     print(z.to("cpu",th.double))


# autograd 操作

x = th.ones(2,2,requires_grad=True)
y = x+2
#print(y)
#print(y.grad_fn)
#print(x.is_leaf,y.is_leaf)
z = y*y*3
out = z.mean()
#print(z,out)


a = th.randn(2,2)
a = (a*3)/(a - 1)
#print(a.requires_grad)
a.requires_grad_(True)
#print(a.requires_grad)
b = (a*a).sum()
#print(b.grad_fn)


# 梯度
# out.backward()
# out2 = x.sum()
# out2.backward()
# out3 = x.sum()
# x.grad.data.zero_()
# out3.backward()
# print(x.grad)

x = th.tensor([1.0,2.0,3.0,4.0],requires_grad=True)
y = th.tensor([[0.1,0.1],[0.2,0.2]],dtype = th.float32)
v = 2*x
z = v.view(2,2)
with th.no_grad():
    c = 4*x
z.backward(y)
print(x.grad,c.requires_grad)