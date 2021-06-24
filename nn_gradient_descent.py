# -*- coding: utf-8 -*-
"""

"""
import random
import numpy as np
np.random.seed(1234)


class NNetwork(object):
    
    def __init__(self, sizes):
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = []
        self.weights = []
        
        # biases
        # for hidden layers and output layer neurons
        # e.g. if 3 hidden layer, 1 output layer then 
        #      return [array([[b1], [b2], [b3]]), array([[b4]])]
        
        for y in sizes[1:]:
            self.biases.append(np.random.randn(y,1))
            
        
        # weights
        # for hidden layers and output layer neurons (randomly assigned values)
        # e.g. if 2 input layer neurons, 3 hidden layer neurons, 1 output layer neuron then  
        #      return [array([[w1 w2], [w3 w4], [w5 w6]]), array([[w7 w8 w9]])]
        
        for i in range(len(sizes)-1):
            self.weights.append(np.random.randn(sizes[i+1],sizes[i]))


if __name__ == "__main__":
    
    # setting up train and test sets
    training_data = []
    for i in range(0,25):
        training_data.append((round(random.random(), 2), random.randint(0, 1)))
    
    test_data = training_data[0:10]
 
    # setup a network with 2 input layers, 3 hidden layers, 1 output layer
    net = NNetwork([2, 3, 1])
