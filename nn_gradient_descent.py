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
        self.delta_b = []
        self.delta_w = []

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

        self.delta_b = [np.zeros(b.shape) for b in self.biases]
        
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
        
        self.delta_w = [np.zeros(w.shape) for w in self.weights] 

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
        layer = 1  #layer 0 is input layer

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

        layer = 1  #layer 0 is input layer

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

        self.activations_list.append(activations)  #input is 0 layer
        self.z_list.append(activations)  #input is 0 layer
        
        layer = 1  #layer 0 is input layer
        
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
        """
        derivative of the sigmoid function
        if a = 1/(1+e^(-z))
           da/dz = -1*(-e^(-z))/(1+e^(-z))^2
                 = e^(-z)/(1+e^(-z))^2
        """
        return np.exp(-1*z)/(1.0+np.exp(-1*z))**2

    def cost_prime(self, A, y):
        """
        Cost = 1/2 * (y - y_pred)^2
        Cost = 1/2 * (y-A)^2 
        dC/dA = 2/2*(y-A)*(-1) = A-y
        """
        
        return (A-y)

    def backprop(self, x, y): 
        
        # feedforward
        #--------------
        self.feedforward(x)


        # backward pass
        #--------------

        # 1. output error (e_L)
        #    e_L = delta_a Cost (.) sigma_prime(Z_L inputs in layer L) --> dC/dZ = dC/dA * dA/dZ,  (.) is the hadamard product  
        #         (how fast cost C is changing with respect to activation A) * how fast activation A is changing with respect to Z 

        error = self.cost_prime(self.output, y) * self.sigmoid_prime(self.z_list[-1])
        
        # 2. Cost relative to bias (dC/dB in layer L = e_L)
        #    dC/dB = dC/dA * dA/dZ * dZ/dB = (e_L) * dZ/dB
        #    Z = W(jk,l).A(k,l-1) + B(j,l) = W.A+B
        #    dZ/dB = 1
        #    dC/dB = (e_L) * dZ/dB = e_L
        
        self.delta_b[-1] = error
        
        # 3. Cost relative to weights (dC/dW in layer L = e_L)
        #    dC/dW = dC/dA * dA/dZ * dZ/dW = (e_L) * dZ/dW
        #    Z = W(jk,l).A(k,l-1) + B(j,l) = W.A+B
        #    dZ/dW = A(k,l-1)
        #    dC/dW = (e_L) * dZ/dW = e_L * A(k,l-1) note: activations in layer l-1 is stored in activations_list[-2]
 
        self.delta_w[-1] = np.dot(error, self.activations_list[-2].transpose())
        
        # 4. Error in layer l in terms of error in layer l+1 (e_l)
        #    dC/dZ_(l) = dC/dZ_(l+1) * dZ_(l+1)/dZ_(l) = dZ_(l+1)/dZ_(l) * dC/dZ_(l+1) = dZ_(l+1)/dZ_(l) * e_(l+1)
        #    Z_(l+1) = W_(l+1) * A_(l) + B_(l+1) = W_(l+1) * sigmoid(Z_(l)) + B_(l+1)
        #    dZ_(l+1)/dZ_(l) = W_(l+1) * d[sigmoid(Z_l)]/dZ_(l)
        #    dC/dZ_(l) = W_(l+1) * d[sigmoid(Z_(l))]/dZ_(l) * e_(l+1)
        #    dC/dZ_(l) = W_(l+1) * e_(l+1) * d[sigmoid(Z_(l))]/dZ_(l)
        for layer in range(2,self.num_layers):
            error = np.dot(self.weights[-1*layer+1].transpose(), error) * self.sigmoid_prime(self.z_list[-1*layer])
            self.delta_b[-1*layer] = error
            self.delta_w[-1*layer] = np.dot(error, self.activations_list[-1*layer-1].transpose())
        
        
        return {'delta_b': self.delta_b, 'delta_w': self.delta_w}



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

    net_back_prop = net.backprop(x, y)
    net.print_activations_map()

    print(f'Backwards Pass:')
    for key, val in net_back_prop.items():
        print(key)
        print(val)
    