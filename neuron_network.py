#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 18:02:48 2017
　　
@author: yuichi
"""
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)
    
def show_step_func_graph():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.title('STEP')
    plt.ylim(-0.1, 1.1) # set y axis range
    plt.show()
    
def test_sigmoid():
    x = np.array([-1.0, 1.0, 2.0])
    ans = sigmoid(x)
    print('0.26894142={0}\n0.73105858={1}\n0.88079708={2}'.
          format(ans[0], ans[1], ans[2]))

def show_sigmoid_graph():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.title('SIGMOID')
    plt.ylim(-0.1, 1.1)
    plt.show()

def show_relu_graph():
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.title('ReLU')
    plt.show()
    
def touch_numpy():
    A = np.array([1, 2, 3, 4])
    print(A)
    print(np.ndim(A))
    print(A.shape)
    print('\n')
    B = np.array([[1, 2], [3, 4], [5, 6]])
    print(B)
    print(np.ndim(B))
    print(B.shape)

def dot_product():
    A = np.array([[1, 2], [3, 4]])
    print('A shape: {0}\nA: \n{1}'.format(A.shape, A))
    B = np.array([[5, 6], [7, 8]])
    print('B shape: {0}\nB: \n{1}'.format(B.shape, B))
    print('A dot B: \n{0}\n'.format(np.dot(A, B)))
    C = np.array([[1, 2, 3], [4, 5, 6]])
    D = np.array([[1, 2], [3, 4], [5, 6]])
    print('{0} \ndot \n{1} \n= \n{2}'.format(C, D, np.dot(C, D)))
    E = np.array([[1, 2], [3, 4], [5, 6]])
    F = np.array([7, 8])
    print('{0} \ndot \n{1} \n= \n{2}'.format(E, F, np.dot(E, F)))

def easy_neuron_network_3_3_3():
    X = np.array([1, 2]) # input layer (layer 0)
    W = np.array([[1, 3, 5], [2, 4, 6]]) # middle layer (layer 1)
    Y = np.dot(X, W) # output (layer 2)
    print(Y)
    
def three_layer_neuron_network():
    # from layer 0 to layer 1
    x = np.array([1.0, 0.5]) # input (layer 0)
    w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # weight
    b1 = np.array([0.1, 0.2, 0.3]) # bias
    a1 = np.dot(x, w1) + b1  # middle (layer 1)
    print(a1)
    # from layer 1 to layer 2

# touch_numpy()
# dot_product()    
show_step_func_graph()
test_sigmoid()
show_sigmoid_graph()
show_relu_graph()
easy_neuron_network_3_3_3()
three_layer_neuron_network()