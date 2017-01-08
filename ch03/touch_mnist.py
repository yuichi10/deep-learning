#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 23:46:14 2017

@author: yuichi
"""

import sys, os
sys.path.append(os.pardir) #
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
                 
def mnist_show():
    (x_train, t_train) , (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    img = x_train[0]
    label = x_train[0]
    print(label)
    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)
    img_show(img)
    

(x_train, t_train) , (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
mnist_show()
