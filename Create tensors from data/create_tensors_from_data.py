# -*- coding: utf-8 -*-
"""Create Tensors from Data.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/194BQWmm0cQB6JgBVgArNDi1mdfen7zd1
"""

import torch
import numpy as np

#Create tensor from list
tensor_list=torch.tensor([1,1,1])
tensor_list

#Create tensor from numpy array
nparray=np.array([[1,1,1],[2,2,2],[3,3,3]])
tensor_array=torch.tensor(nparray)
tensor_array

#Create tensor with zeros and ones
tensor_zeros=torch.zeros(2,2)
tensor_ones=torch.ones(2,2)
print(tensor_zeros)
print(tensor_ones)

#Create random tensors
random_tensor=torch.rand(3,3)  #Uniform Distribution between 0 and 1
normal_tensor=torch.randn(3,3) #Normal Distribution with mean 0 and variance 1
print(random_tensor)
print(normal_tensor)

#Specify its type
tensor_int=torch.tensor([1,1,1],dtype=torch.int)
tensor_float=torch.tensor([1.5,1.7,1.9],dtype=torch.float32)
print(tensor_int)
print(tensor_float)

#Create tensors using GPU if it is avaliable
tensor_gpu=torch.tensor([1,1,1],device='cuda')
print(tensor_gpu)