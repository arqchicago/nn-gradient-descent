# Gradient Descent in Neural Networks
This project implements gradient descent algorithm from scratch to train a Neural Network. Gradient Descent is used in Neural Networks 
to optimize the cost function by tuning hyperparameters; including the weights and biases of the network. 


## Blog 
To Be Posted

## NNetwork Class
In this program, a NNetwork class is defined that sets up biases, weights, input, output and hidden layers of a neural network. User 
can setup their choice of the number of hidden layers. 

## First Simulation
In the first simulation, a 3-layer 2x3x2 network is defined: 2 layers in the input layer, 3 in the hidden layer and 2 in the output
layer. Initially, biases and weights are assigned randomly.

## Feedforward
In this step, activations are computed for each neuron in the hidden layer and eventually the output layer. In each layer, starting from
the first hidden layer, each neuron is picked and its activation is computed. This activation is the value of sigmoid function that
takes input z. This input z is the sum of bias of a neuron in current layer and sum of the products of weights of links between this neuron 
and neurons from the previous layer with the activation of the neurons in the previous layer. In other words, 
z = sum_k [w(layer l)_j_k  a(layer l-1)_k] + b(layer l)_j where l is the current layer, l-1 is the previous layer, w_j_k is the weight
of connection between neuron j in layer l and neuron k in layer l-1, a_k is the activation of neuron k in layer l-1 and b_j is the bias
of neuron in layer j. 
Sigmoid of z is defined as sigmoid(z) = 1/(1-e^(-z))
In the feedforward step, each neuron in each layer is picked, sigmoid of z is computed where z is calculated using w weights and a activations.
This process is repeated until the entire network is traversed. 