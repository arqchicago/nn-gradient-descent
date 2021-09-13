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
        self.activations = {}
        self.bias_map = {}
        self.weight_map = {}
        self.output = [0]
        self.bias_list = []
        self.weights_list = []
        self.z_list = []
        self.activations_list = []
        
        """ 
        biases
        for hidden layers and output layer neurons
        e.g. if 3 hidden layer neurons, 1 output layer neuron then 
             return [array([[b1], [b2], [b3]]), array([[b4]])]
        #
        #    where b1 is bias of neuron 1 in layer 2
        #          b2 is bias of neuron 2 in layer 2 and so on...
        """

        for y in sizes[1:]:
            self.biases.append(np.random.randn(y,1).round(2))

        """
        # weights
        # for hidden layers and output layer neurons (randomly assigned values)
        # e.g. if 2 input layer neurons, 3 hidden layer neurons, 1 output layer neuron then  
        #      return [array([[w1 w2], [w3 w4], [w5 w6]]), array([[w7 w8 w9]])]
        #
        #      where w1 is weight of link between neuron 1 in layer 2 and neuron 1 in layer 1
        #            w2 is weight of link between neuron 1 in layer 2 and neuron 2 in layer 1
        #            w3 is weight of link between neuron 2 in layer 2 and neuron 1 in layer 1 and so on...
        """

        for i in range(len(sizes)-1):
            self.weights.append(np.random.randn(sizes[i+1],sizes[i]).round(2))

    def get_num_layers(self):
        return self.num_layers
  
    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

    def update_bias_map(self):
        """
        this will return detailed information on biases of neurons in the network
        e.g. Neuron O(layer 2, neuron 1) --> biase_info[layer_num][neuron_num] = biase_info[2][1]
        """
        layer = 2  #layer 1 is input layer

        for biases in self.biases:
            layer_id = 'layer_'+str(layer)
            self.bias_map[layer_id] = {}
            neuron = 1
            for biase in biases:
                neuron_id = 'neuron_'+str(neuron)
                self.bias_map[layer_id][neuron_id] = biase[0]
                neuron+=1
            layer+=1
            
    def print_bias_map(self):
        # print the bias map
        print('='*50)
        print('{:<5} {:<12} {:<12} {:>12}'.format(' ', 'layer', 'neuron', 'bias'))
        print('-'*50)
        for layer, neuron_biases in self.bias_map.items():
            for neuron, bias in neuron_biases.items():
                print('{:<5} {:<12} {:<12} {:>12}'.format(' ', layer, neuron, bias))

        print('='*50)
        print()

    def update_weight_map(self):
        """
        this will return detailed information on the weights of inputs to neurons in the network
        e.g. Neuron O(layer 2, neuron 1) -- O(layer 3, neuron 2)  --> layer_2, 
        """

        layer = 2  #layer 1 is input layer

        for layer_weights in self.weights:
            layer_id = 'layer_'+str(layer)
            self.weight_map[layer_id] = {}
            neuron = 1

            for neuron_weights in layer_weights:
                neuron_id = 'neuron_'+str(neuron)
                self.weight_map[layer_id][neuron_id] = {}
                neuron_prev = 1

                for weight in neuron_weights:
                    neuron_prev_id = 'neuron__'+str(neuron_prev)
                    self.weight_map[layer_id][neuron_id][neuron_prev_id] = weight
                    neuron_prev += 1  
                neuron += 1
            layer+=1
            
    def print_weight_map(self):
        print('='*70)
        print('{:<5} {:<12} {:<12} {:>12} {:>12}'.format(' ', 'layer', 'neuron', 'prev neuron', 'weight'))
        print('-'*70)
        
        # print weights
        for layer, neuron_weights in self.weight_map.items():
            for neuron, weights in neuron_weights.items():
                for neuron_prev_layer, weight in weights.items():
                    print('{:<5} {:<12} {:<12} {:>12} {:>12}'.format(' ', layer, neuron, neuron_prev_layer, weight))
        print('='*70)
        print()

    def feedforward(self, activations):
        """
        Return the output of the network if activations is input.
        traverse through the neurons of each layer and calculate activations until the output layer neurons
        z = w*activations + b
        activations = 1/(1+exp(-z))
        output of neurons in one layer is inputs to the neurons in the next layer
        """

        self.activations_list.append(activations)
        self.z_list.append(activations)
        
        layer = 2  #layer 1 is input layer
        
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activations)+bias
            activations = self.sigmoid(z)
            
            layer_id = 'layer_'+str(layer)
            self.activations[layer_id] = {}
            neuron = 1
            for activation in activations:
                neuron_id = 'neuron_'+str(neuron)
                self.activations[layer_id][neuron_id] = activation[0]
                neuron += 1
            layer += 1
            self.z_list.append(z)
            self.activations_list.append(activations)
        self.output = activations
        
    def get_output(self):
        return self.output
        
    def get_zs(self):
        return self.z_list
            
    def print_activations_map(self):
        print('='*50)
        print('{:<5} {:<12} {:<12} {:>12}'.format(' ', 'layer', 'neuron', 'activation'))
        print('-'*50)
        
        # print weights
        for layer, neuron_activations in self.activations.items():
            for neuron, activation in neuron_activations.items():
                print('{:<5} {:<12} {:<12} {:>12}'.format(' ', layer, neuron, round(activation, 8)))

        print('='*50)
        print()
        
    def sigmoid(self, z):
        """sigmoid function"""
        return 1.0/(1.0+np.exp(-1*z))

    def sigmoid_prime(self, z):
        """derivative of the sigmoid function"""
        return np.exp(-1*z)/(1.0+np.exp(-1*z))**2

    def cost_prime(self, output, y):
        """derivative of the cost function:  output - y """
        return (output-y)

    def backprop(self, x, y):
       
        # feedforward
        self.feedforward(x)
        
        # backward pass
        # - output error (e_L)
        #   -- e_L = delta_a Cost (.) sigma_prime(Z_L inputs in layer L) --> dC/dA * dA/dZ  
        #           (how fast cost C is changing with respect to activation A) * how fast activation A is changing with respect to Z 
        output_error = self.cost_prime(self.output, y) * self.sigmoid_prime(self.z_list[-1])
        

        return output_error



if __name__ == "__main__":

    # NETWORK 1
    # setup a network with 2 input neurons, 3 hidden neurons, 2 output neuron
    net = NNetwork([2, 3, 2])
    num_net_layers = net.get_num_layers()
    biases_net = net.get_biases()
    weights_net = net.get_weights()

    print('') 
    print(f'Network #1')   
    print(f'{"-"*90}')   
    print(f'number of layers: \n{num_net_layers}\n')
    print(f'biases \n{biases_net}\n')
  
    net.update_bias_map()
    net.print_bias_map()
     
    net.update_weight_map()
    net.print_weight_map()
        
    x = np.random.rand(2,1).round(2)
    y = np.random.rand(2,1).round(2)
    print(f'\ninput')
    print(x)
    #net_output = net.feedforward(x)
    net_back_prop = net.backprop(x, y)
    net.print_activations_map()
    y_pred = net.get_output()

    d_output = np.exp(-1*y_pred)/(1.0+np.exp(-1*y_pred))**2
    cost_prime = (y_pred-y)
    print(y_pred)
    print(y)
    print(d_output)
    print(cost_prime)
    print(net_back_prop)
    print(cost_prime*d_output)
    