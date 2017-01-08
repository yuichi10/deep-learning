#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 18:11:00 2017

@author: yuichi
"""
import numpy as np

def improvable_softmax(x):
    exp_x = np.exp(x)
    exp_sum = np.sum(exp_x)
    y = exp_x / exp_sum
    return y

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(a - c) # try not to happen stack over flow
    exp_sum = np.sum(exp_x)
    y = exp_x / exp_sum
    return y
    
a = np.array([0.3, 2.9, 4.0])
print(improvable_softmax(a))
a = np.array([1010, 1000, 990])
print(improvable_softmax(a))
print(softmax(a))
print('sum of softmax: {}'.format(np.sum(softmax(a))))