#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Create by viekie
# Create on 2021/8/31

from __future__ import print_function
import tensorflow as tf

print(tf.__version__)
print('gpu is available : ' + str(tf.test.is_gpu_available()))

x = tf.constant(5, 3)
print(x)
