#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/8/31

from __future__ import print_function
import torch

x = torch.rand((5, 3), requires_grad=False)
print(x)
y = torch.rand((5, 3), requires_grad=True)
print(y)
z = x + y
print(z)
print(z.grad_fn)
a = torch.add(torch.matmul(x.t(), y), 2)
print(a)

print('#' * 50)

a = torch.randn(2, 2)
print(a)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
print(a)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

print('#' * 50)

b.backward()
print(a.grad)

print('#' * 50)

x = torch.randn(3, requires_grad=True)
print(x.grad)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

print('#' * 50)
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)