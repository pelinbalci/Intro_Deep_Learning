import numpy as np
import torch

a = np.random.rand(1, 2)

print('a is a random variable: ', a)
print('a type:', a.dtype)

b = torch.from_numpy(a)

print('b is coming from a: ', b)
print('b type:', b.dtype)

c = b.numpy() # b type is not changed with this operation. b is still torch.

print('c is numpy version of b: ', c)
print('c type:', c.dtype)


print('b type is not change after add numpy(): ', b.dtype)


b.mul_(2)

print('b, after multiply it by 2: ', b)
print('b type:', b.dtype)

print('a, after multiply b by 2: ', a)
print('a type:', a.dtype)

print('c, after multiply b by 2: ', c)
print('c type:', c.dtype)

