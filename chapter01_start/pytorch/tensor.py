#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/8/31

from __future__ import print_function
import torch
import numpy as np

print(torch.__version__)
print('gpu is available: ' + str(torch.cuda.is_available()))

x = torch.empty((5, 3))
y = torch.rand((5, 3))
z = torch.randn((5, 3))
a = torch.rand_like(
    torch.Tensor(
        [[1, 2, 3.0],
         [4, 5, 6]]
    )
)

b = a.new_empty((6, 4))
c = b.new_ones((4, 6))
print(x)
print(y)
print(z)
print(a)
print(b)
print(c)
print(c.size())

print('###############################################')

p = y + z
q = torch.add(y, z)
print(p)
print(q)
y.add_(z)
print(y)

print('###############################################')

print(y.size())
print(y.size()[1:])
m = y.view(-1, 5)
print(m)

print('###############################################')

d = m.numpy()
print(d)
m.add_(m)
print(d)

print('###############################################')

e = np.ones((5, 3))
f = torch.from_numpy(e)
print(e)
print(f)

print('###############################################')
if torch.cuda.is_available():
    device = torch.device('cuda')
    h = torch.ones((3, 5))
    g = torch.zeros((3, 5), device=device)
    # k = h + g  Error Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
    print(h)
    print(g)
    h_slash = h.to(device=device)
    l = h_slash + g
    print(l)
