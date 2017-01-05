#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 17:23:47 2017

@author: yuichi
"""

import numpy as np


def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
        
def AND_BIAS(x1, x2):
    weight = np.array([0.5, 0.5]) # weight
    bias = -0.7 # bias
    
    input_val = np.array([x1, x2]) # input values
    tmp = np.sum(weight * input_val) + bias
    if tmp <= 0:
        return 0
    else:
        return 1
        
def NAND(x1, x2):
    weight = np.array([-0.5, -0.5])
    bias = 0.7
    
    input_val = np.array([x1, x2])
    tmp = np.sum(weight * input_val) + bias
    if tmp <= 0:
        return 0
    else:
        return 1
        
def OR(x1, x2):
    weight = np.array([0.5, 0.5])
    bias = -0.4
    
    input_val = np.array([x1, x2])
    tmp = np.sum(weight * input_val) + bias
    if tmp <= 0:
        return 0
    else:
        return 1
        
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
        
print('AND PERCEPTRON')
print('(0, 0) => {}'.format(AND(0, 0)))
print('(1, 0) => {}'.format(AND(1, 0)))
print('(0, 1) => {}'.format(AND(0, 1)))
print('(1, 1) => {}'.format(AND(1, 1)))
print('AND PERCEPTRON BY BIAS')
print('(0, 0) => {}'.format(AND_BIAS(0, 0)))
print('(1, 0) => {}'.format(AND_BIAS(1, 0)))
print('(0, 1) => {}'.format(AND_BIAS(0, 1)))
print('(1, 1) => {}'.format(AND_BIAS(1, 1)))

print('\nNAND PERCEPTRON')
print('(0, 0) => {}'.format(NAND(0, 0)))
print('(1, 0) => {}'.format(NAND(1, 0)))
print('(0, 1) => {}'.format(NAND(0, 1)))
print('(1, 1) => {}'.format(NAND(1, 1)))

print('\nOR PERCEPTRON')
print('(0, 0) => {}'.format(OR(0, 0)))
print('(1, 0) => {}'.format(OR(1, 0)))
print('(0, 1) => {}'.format(OR(0, 1)))
print('(1, 1) => {}'.format(OR(1, 1)))

print('\nXOR PERCEPTRON')
print('(0, 0) => {}'.format(XOR(0, 0)))
print('(1, 0) => {}'.format(XOR(1, 0)))
print('(0, 1) => {}'.format(XOR(0, 1)))
print('(1, 1) => {}'.format(XOR(1, 1)))
    