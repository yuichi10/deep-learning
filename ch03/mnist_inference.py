#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 01:49:53 2017

@author: yuichi
"""

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
import pickle

class MnistInference:
    def __init__(self):
        self.set_data()
        self.init_network()
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        c = np.max(x)
        exp_x = np.exp(x - c)
        exp_sum = np.sum(exp_x)
        return exp_x / exp_sum
        
    def set_data(self):
        (x_train, t_train) , (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
        self.x_test = x_test
        self.t_test = t_test
        
    def init_network(self):
        with open("sample_weight.pkl", 'rb') as f:
            network = pickle.load(f)
        
        self.network = network
        
    def predict(self, network, x):
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = self.sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = self.softmax(a3)
        return y
    
    def get_result(self):
        accuracy_cnt = 0
        for i in range(len(self.x_test)):
            y = self.predict(self.network, self.x_test[i])
            p = np.argmax(y)
            if p == self.t_test[i]:
                accuracy_cnt += 1
        print('Accuracy:' + str(float(accuracy_cnt) / len(self.t_test)))
        

class MnistInferenceBatch(MnistInference):
    def __init__(self):
        super().__init__()
    
    def get_result(self):
        batch_size = 100
        accuracy_cnt = 0
        
        for i in range(0, len(self.x_test), batch_size):
            x_batch = self.x_test[i:i+batch_size]
            y_batch = self.predict(self.network, x_batch)
            p = np.argmax(y_batch, axis=1)
            accuracy_cnt += np.sum(p == self.t_test[i:i+batch_size])
        print('Accuracy:' + str(float(accuracy_cnt) / len(self.t_test)))
        
        
neuralnet = MnistInference()
neuralnet.get_result()

neuralnetBatch = MnistInferenceBatch()
neuralnetBatch.get_result()
