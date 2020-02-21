import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = torch.tensor([2, -3.4])
true_b = torch.tensor([4.2])
features = torch.randn(num_examples,num_inputs,dtype=torch.float32)
labels = torch.matmul(features,true_w) + true_b
labels += torch.tensor(np.random.normal(0,0.01,size = labels.size()),dtype = torch.float32)

# print(features[0],labels[0])
#生成features labels的散点图，观察线性关系

def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize = (3.5,2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

set_figsize()

# plt.scatter(features[:,1].numpy(),labels.numpy(),1)
# # plt.show()

# 读取数据
def data_iter(batch_size,features,labels):
    num_exapmels = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size,num_examples)])
        yield features.index_select(0,j),labels.index_select(0,j)

# for X,y in data_iter(10,features,labels):
#     print(X,y)
#     break
# 初始化模型参数
w = torch.tensor(np.random.normal(0,0.01,size = (num_inputs,1)),dtype = torch.float32)
b = torch.zeros(1,dtype = torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 定义模型
def linear(X,w,b):
    return torch.mm(X,w) + b

# 定义损失函数
def square_loss(y_hat,y):
    return (y_hat - y.view(y_hat.size()))**2 / 2

def sgd(params,lr,batch_size):
    for param in params:
        param.data -= lr * param.grad/batch_size
        # 这里param.grad求出的梯度是batch_size的梯度和，求平均后更新权重


# 训练模型
# 超参数
lr = 0.03
batch_size = 10
num_epochs = 3
net = linear
loss = square_loss
for epoch in range(num_epochs):
    for X,y in data_iter(10,features,labels):
        l = loss(net(X,w,b),y).sum()
        l.backward()
        sgd([w,b],lr,batch_size)

        # 梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()

    train_l = loss(net(features,w,b),labels).mean()
    print('epoch %d,loss %f'%(epoch + 1,train_l.item()))

# 输出训练结果
print(true_w,'\n',w)
print(true_b,'\n',b)