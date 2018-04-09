"""A Simple Neural Network implementation,
    largely lifted from https://neuralnetworksanddeeplearning.com/"""

import random
import cleansing
import numpy as np

EPSILON = 0.4

class Network(object):
    """Implementation of a simple neural net."""

    def __init__(self, sizes):
        """Constructor:
        takes the sizes of each layer of the neural network
        and makes the random weights and biases"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.reset()

    def reset(self):
        """Resets weights and biases to random values."""
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, example):
        """Function to ``run`` the network with an input vector of a"""
        if len(example.shape) == 1:
            example = reguralize_input(example)
        for b, w in zip(self.biases, self.weights):
            example = sigmoid(np.dot(w, example)+b)
        return example

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Training function:
        ``training_data`` is a list of tuples ``(input, desired_output)``
        where input and desired_output are numpy arrays. test_data follows the same format"""
        reshape(training_data)
        if test_data:
            reshape(test_data)
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Train on a small subset of data to increase training speed"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, example, expectation):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_example.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = example
        activations = [example] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b # output should be a vector
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], expectation) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, expected):
        """Return the vector of partial derivatives partial C_x /
        partial a for the output activations."""
        return output_activations - expected

    def evaluate(self, test_data):
        """Calculate how many test data points are correct"""
        test_results = [(self.feedforward(x), y)
                        for (x, y) in test_data]
        print(sum(abs(x - y) for x, y in test_results))
        return sum(int(abs(x - y) < EPSILON) for (x, y) in test_results)

def reshape(data):
    """Makes sure that all elements in `data` are column vectors."""
    for i, entry in enumerate(data):
        ins, out = entry
        if len(ins.shape) == 1:
            ins = reguralize_input(ins)
        if len(out.shape) == 1:
            out = reguralize_input(out)
        data[i] = (ins, out)

def column(lst):
    """Transforms a standard Python list into a numpy column vector."""
    return cleansing.matrix_from(lst)

def uncolumn(vec):
    """Transforms a numpy column vector to standard Python list."""
    return vec.T.tolist()[0]

def reguralize_input(vec):
    """Input a simple numpy vector, and outputs a column vector."""
    # input v is a vector
    return np.array([vec.tolist()]).T

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
