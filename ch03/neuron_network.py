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
    

class OwnThreeLayerNeuronNetork:
    def __init__(self):
        self.w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # weight layer 1
        self.b1 = np.array([0.1, 0.2, 0.3]) # bias layer 1
        self.w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) # weight layer 2
        self.b2 = np.array([0.1, 0.2])
        self.w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
        self.b3 = np.array([[0.1, 0.2]])
    
    def show_result(self, **args):
        if 'title' in args:
            print(args['title']) 
        if 'input_value' in args:
            print('input vaule: {}'.format(args['input_value']))
        if 'use_weight_value' in args:
            print('use weight value: {}'.format(args['use_weight_value']))
        if 'use_sigmoid_value' in args:
            print('use sigmoid value: {}'.format(args['use_sigmoid_value']))
        if 'use_identity_value' in args:
            print('use identity function value: {}'.format(args['use_identity_value']))
        print('\n')
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def identity_function(self, x):
        return x
    
    def get_data(self, x):
        self.show_result(title='layer 0', input_value=x)
        z1 = self.layer1(x)
        z2 = self.layer2(z1)
        y = self.layer3(z2)
        return y
    
    def layer1(self, x):
        a1 = np.dot(x, self.w1) + self.b1
        z1 = self.sigmoid(a1)
        self.show_result(title='layer 1', use_weight_value=a1, use_sigmoid_value=z1)
        return z1
    
    def layer2(self, z1):
        a2 = np.dot(z1, self.w2) + self.b2
        z2 = self.sigmoid(a2)
        self.show_result(title='layer 2', use_weight_value=a2, use_sigmoid_value=z2)
        return z2        
    
    def layer3(self, z2):
        a3 = np.dot(z2, self.w3) + self.b3
        y = self.identity_function(a3)
        self.show_result(title='layer 3', use_weight_value=a3, use_identity_value=a3)
        return y
        
        
class ThreeLayerNeuronNetwork:
    def __init__(self):
        self.init_network()
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def identity_function(self, x):
        return x
        
    def init_network(self):
        network = {}
        network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        network['b1'] = np.array([0.1, 0.2, 0.3])
        network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        network['b2'] = np.array([0.1, 0.2])
        network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
        network['b3'] = np.array([[0.1, 0.2]])
        return network
    
    def forward(self, network, x):
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = self.sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        z3 = self.sigmoid(a3)
        y = self.identity_function(z3)
        return y
    
    def start_network(self):
        network = self.init_network()
        x = np.array([1.0, 0.5])
        y = self.forward(network, x)
        print(y)
        
        
# touch_numpy()
# dot_product()    
show_step_func_graph()
test_sigmoid()
show_sigmoid_graph()
show_relu_graph()
easy_neuron_network_3_3_3()

threelayer = OwnThreeLayerNeuronNetork()
threelayer.get_data(np.array([1.0, 0.5]))

threelayer = ThreeLayerNeuronNetwork()
threelayer.start_network()