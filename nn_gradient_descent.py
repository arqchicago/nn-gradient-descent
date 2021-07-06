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

    def get_num_layers(self):
        return self.num_layers
        
    def get_weights(self):
        return self.weights
        
    def get_biases(self):
        return self.biases



if __name__ == "__main__":
    
    # setting up train and test sets
    training_data = []
    for i in range(0,25):
        training_data.append((round(random.random(), 2), random.randint(0, 1)))
    
    test_data = training_data[0:10]
 
    # setup a network with 2 input layers, 3 hidden layers, 1 output layer
    net = NNetwork([2, 3, 1])
    num_net_layers = net.get_num_layers()
    biases_net = net.get_biases()
    weights_net = net.get_weights()

    print(f'> Network #1\n')    
    print(f'> number of layers: \n{num_net_layers}\n')
    print(f'> biases: \n{biases_net}\n')
    print(f'> weights: \n{weights_net}')


    # setup a network with 3 input layers, 4 hidden layers, 2 output layer
    net2 = NNetwork([3, 4, 2])
    num_net2_layers = net2.get_num_layers()
    biases_net2 = net2.get_biases()
    weights_net2 = net2.get_weights()

    print(f'> Network #2\n')   
    print(f'> number of layers: \n{num_net2_layers}\n')
    print(f'> biases: \n{biases_net2}\n')
    print(f'> weights: \n{weights_net2}')
