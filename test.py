#导入需要的包和模块
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..")# 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

mnist_train = torchvision.datasets.FashionMNIST(root = '~/Datasets/FashionMNIST',train=True,download=True,transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST',train=False,download=True,transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train),len(mnist_test))