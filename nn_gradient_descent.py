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

        
        """ 
        biases
        for hidden layers and output layer neurons
        e.g. if 3 hidden layer, 1 output layer then 
             return [array([[b1], [b2], [b3]]), array([[b4]])]
        """

        for y in sizes[1:]:
            self.biases.append(np.random.randn(y,1).round(2))


        """
        # weights
        # for hidden layers and output layer neurons (randomly assigned values)
        # e.g. if 2 input layer neurons, 3 hidden layer neurons, 1 output layer neuron then  
        #      return [array([[w1 w2], [w3 w4], [w5 w6]]), array([[w7 w8 w9]])]
        """

        for i in range(len(sizes)-1):
            self.weights.append(np.random.randn(sizes[i+1],sizes[i]).round(2))


    def get_num_layers(self):
        return self.num_layers

  
    def get_weights(self):
        return self.weights


    def get_weight_details(self):
        """
        this will return detailed information on the weights of inputs to neurons in the network
        e.g. Neuron O(layer 2, neuron 1) -- O(layer 3, neuron 2)  --> layer_2, 
        """

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
        """
        this will return detailed information on biases of neurons in the network
        e.g. Neuron O(layer 2, neuron 1) --> biase_info[layer_num][neuron_num] = biase_info[2][1]
        """
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


    def sigmoid(self, z):
        """sigmoid function"""
        return 1.0/(1.0+np.exp(-1*z))


    def d_sigmoid(self, z):
        """derivative of the sigmoid function"""
        return np.exp(-1*z)/(1.0+np.exp(-1*z))**2

    def feedforward(self, activations):
        """
        Return the output of the network if activations is input.
        traverse through the neurons of each layer and calculate activations until the output layer neurons
        z = w*activations + b
        activations = 1/(1+exp(-z))
        output of neurons in one layer is inputs to the neurons in the next layer
        """
        
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
        return activations


    def stochastic_gradient_descent(self, epochs, batch_size, training_data):
        """
        This method will use backpropagation on mini-batches of size batch_size each to update weights and biases. 
        """

        n = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
        
            #for batch in batches:
            #    for x, y in batch:
            #        print(">> (",x,y,")")
        return batches

    def get_activation_details(self):
        """
        Return activations of all neurons in the network.
        """
        return self.activations



if __name__ == "__main__":
    
    epochs = 10
    batch_size = 5
    
    # setting up train and test sets
    training_data = []
    for i in range(0,100):
        training_data.append((round(random.random(), 2), random.randint(0, 1)))
    
    test_data = training_data[0:10]
    training_data = training_data[10:100]



    # NETWORK 1
    # setup a network with 2 input neurons, 3 hidden neurons, 2 output neuron
    net = NNetwork([2, 3, 2])
    num_net_layers = net.get_num_layers()
    biases_net = net.get_biases()
    weights_net = net.get_weights()
    biases_net_details = net.get_biase_details()
    weights_net_details = net.get_weight_details()

    print('') 
    print(f'> Network #1')    
    print(f'> number of layers: \n{num_net_layers}\n')
    print(f'> biases: \n{biases_net}\n')
    print(f'> weights: \n{weights_net}')
    
    # print biases
    for layer, neuron_biases in biases_net_details.items():
        for neuron, biase in neuron_biases.items():
            print(f'layer = {layer}, neuron = {neuron}:  biase = {biase}')


    # print weights
    for layer, neuron_weights in weights_net_details.items():
        for neuron, weights in neuron_weights.items():
            for neuron_prev_layer, weight in weights.items():
                print(f'layer = {layer}, neuron = {neuron}, prev neuron = {neuron_prev_layer}:  weight = {weight}')

    batches = net.stochastic_gradient_descent(epochs, batch_size, training_data)
    
    #print(batches)
    #print(len(batches))
    a = np.random.rand(2,1).round(2)
    net_output = net.feedforward(a)
    activations = net.get_activation_details()

    print(f'input = {a}')
    print(f'network output = {net_output}')
    
    # print activations
    for layer, neuron_activations in activations.items():
        for neuron, activation in neuron_activations.items():
            print(f'layer = {layer}, neuron = {neuron}:  activation = {activation}')



    
    
    '''
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
    
    # print biases
    for layer, neuron_biases in biases_net2_details.items():
        for neuron, biase in neuron_biases.items():
            print(f'layer = {layer}, neuron = {neuron}:  biase = {biase}')    

    for layer, neuron_weights in weights_net2_details.items():
        for neuron, weights in neuron_weights.items():
            for neuron_prev_layer, weight in weights.items():
                print(f'layer = {layer}, neuron = {neuron}, prev neuron = {neuron_prev_layer}:  {weight}')

    a = np.random.rand(3,1).round(2)
    net2_output = net2.feedforward(a)
    activations = net2.get_activation_details()


    # print activations
    for layer, neuron_activations in activations.items():
        for neuron, activation in neuron_activations.items():
            print(f'layer = {layer}, neuron = {neuron}:  activation = {activation}')

    print(f'network 2 output = {net2_output}')
    
    '''