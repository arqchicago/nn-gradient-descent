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


    def get_weight_details(self):
        # this will return detailed information on the weights of inputs to neurons in the network
        # e.g. Neuron O(layer 2, neuron 1) -- O(layer 3, neuron 2)  --> layer_2, 
        weight_info = {}
        layer = 2  #layer 1 is input layer

        for layer_weights in self.weights:
            layer_id = 'layer_'+str(layer)
            weight_info[layer_id] = {}
            neuron = 1

                
            for neuron_weights in layer_weights:
                neuron_id = 'neuron_'+str(neuron)
                weight_info[layer_id][neuron_id] = {}
                neuron_prev = 1

                for weight in neuron_weights:
                    neuron_prev_id = 'neuron__'+str(neuron_prev)
                    weight_info[layer_id][neuron_id][neuron_prev_id] = weight
                    neuron_prev += 1
                    
                neuron += 1
            layer+=1
            
        return weight_info


    def get_biases(self):
        return self.biases


    def get_biase_details(self):
        # this will return detailed information on biases of neurons in the network
        # e.g. Neuron O(layer 2, neuron 1) --> biase_info[layer_num][neuron_num] = biase_info[2][1]
        biase_info = {}
        layer = 2  #layer 1 is input layer

        for biases in self.biases:
            layer_id = 'layer_'+str(layer)
            biase_info[layer_id] = {}
            neuron = 1
            for biase in biases:
                neuron_id = 'neuron_'+str(neuron)
                biase_info[layer_id][neuron_id] = biase[0]
                neuron+=1
            layer+=1
            
        return biase_info


if __name__ == "__main__":
    
    # setting up train and test sets
    training_data = []
    for i in range(0,25):
        training_data.append((round(random.random(), 2), random.randint(0, 1)))
    
    test_data = training_data[0:10]
 
    # setup a network with 2 input neurons, 3 hidden neurons, 1 output neuron
    net = NNetwork([2, 3, 1])
    num_net_layers = net.get_num_layers()
    biases_net = net.get_biases()
    weights_net = net.get_weights()
    biases_net_details = net.get_biase_details()
    weights_net_details = net.get_weight_details()

    print('') 
    print(f'> Network #1')    
    print(f'> number of layers: \n{num_net_layers}\n')
    print(f'> biases: \n{biases_net}\n')
    print(f'> biases (details): \n{biases_net_details}')
    print(f'> weights: \n{weights_net}')
    
    for layer, neuron_weights in weights_net_details.items():
        for neuron, weights in neuron_weights.items():
            for neuron_prev_layer, weight in weights.items():
                print(f'layer = {layer}, neuron = {neuron}, prev neuron = {neuron_prev_layer}:  {weight}')



    # setup a network with 3 input neurons, 2 hidden layers with 4 neurons each, 2 output neurons
    net2 = NNetwork([3, 4, 4, 2])
    num_net2_layers = net2.get_num_layers()
    biases_net2 = net2.get_biases()
    weights_net2 = net2.get_weights()
    biases_net2_details = net2.get_biase_details()
    weights_net2_details = net2.get_weight_details()

    print('') 
    print(f'> Network #2\n')   
    print(f'> number of layers: \n{num_net2_layers}\n')
    print(f'> biases: \n{biases_net2}\n')
    print(f'> biases (details): \n{biases_net2_details}')
    print(f'> weights: \n{weights_net2}')
    

    for layer, neuron_weights in weights_net2_details.items():
        for neuron, weights in neuron_weights.items():
            for neuron_prev_layer, weight in weights.items():
                print(f'layer = {layer}, neuron = {neuron}, prev neuron = {neuron_prev_layer}:  {weight}')
