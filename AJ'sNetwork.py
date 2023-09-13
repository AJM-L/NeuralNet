#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 19:29:20 2023

@author: ajmatheson-lieber
"""

import random
import numpy as np
from collections import deque

class Network (object):
    
    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.layers = len(sizes)
        self.sizes = sizes
        self.bias = []
        self.weights = []
        for i in sizes[1:]:
            self.bias.append(np.random.randn(i, 1))
        for x, y in zip(sizes[:-1], sizes[1:]):
            self.weights.append(np.random.randn(y, x))
        
    def feedforward(self, a):
        """Return the output of the network if "a" is input.
        a′=σ(wa+b)"""
        for b, w in zip(self.bias, self.weights):
            z = np.dot(w, a) + b
            a = sigmoid(z)
        return a
    
    def calculate_ZA (self, a):
        Z = []
        A = []
        for b, w in zip(self.bias, self.weights):
            z = np.dot(w, a) + b
            a = sigmoid(z)
            Z.append(z)
            A.append(a)
        return Z, A

    def SGD (self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:
            test_len = len(test_data)
        training_len = len(training_data)
        for i in range(epochs):
        # shuffle the training data
            random.shuffle(training_data)
        #partition into mini-batches
            mini_batches = []
            for k in range(0, training_len, mini_batch_size):
                mini_batches.append(training_data[k:k+ mini_batch_size])
            #apply gradient descent for each batch using update_mini_batch function
            #which will update the weights and biases accordingly
            for batch in mini_batches:
                self.update_mini_batch(batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), test_len))
            else:
                print(f"Epoch: {i}")
        
        
        
        
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        #create empty (zeros) arrays for the weights and biases
        delta_w = []
        for w in self.weights:
            delta_w.append(np.zeros(np.shape(w)))
        delta_b = []
        for b in self.bias:
            delta_b.append(np.zeros(np.shape(b)))
        #iterate through the batches, calculating the gradient for each training exampole
        for x, y in mini_batch:
            grad_b, grad_w= self.backprop(x, y)
         # add the gradient of the example to the bias and weight arrays
            delta_w = [dw + gw for dw, gw in zip(delta_w, grad_w)]
            delta_b = [db + gb for db, gb in zip(delta_b, grad_b)]
        #update the weights and biases using the generated gradient
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, delta_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.bias, delta_b)]
        
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        Z, A = self.calculate_ZA(x)
        grad_Ca = A[-1]-y
        E = [[]] * self.layers
        E[-1] = np.multiply(grad_Ca, sigmoid_prime(Z[-1]))
        nabla_b[-1] = E[-1]
        nabla_w[-1] = np.dot(E[-1], A[-2].transpose())
        for layer in range(2, self.layers):
            E[-layer] = np.multiply(np.dot(self.weights[-layer+1].transpose(), E[-layer+1]), sigmoid_prime(Z[-layer]))
            nabla_b[-layer] = E[-layer]
            nabla_w[-layer+1] = np.dot(E[-layer+1], A[-layer].transpose())
        return (nabla_b, nabla_w)
                
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
def sigmoid (z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

"""
MyNet = Network([784, 200, 80, 10])
MyNet.SGD(list(zip(x_train, y_train)), 3, 100, 0.1)
MyNet.evaluate(list(zip(x_test, y_test)))

MyNet = Network([784, 12, 12, 12, 10])
MyNet.SGD(list(zip(x_train, y_train)), 30, 1000, 2, list(zip(x_test, y_test)))
"""
