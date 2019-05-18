#!/usr/bin/env python
# coding: utf-8
"""

Author: Raman Kahlon
Last modified: Apr. 25, 2019

"""

from classifier import BinaryClassifier
from utils import get_feature_vectors
import numpy as np
import random

class SGDHinge(BinaryClassifier):
    
    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.weights = np.zeros(args.vocab_size)
        self.bias = 0
        self.updates = 0
        self.rate = args.lr_sgd
        self.num_iter = args.num_iter
        self.vocab_size = args.vocab_size
        self.bin_feats = args.bin_feats
        #raise NotImplementedError
        
    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        tr_size = len(train_data[0])
        indices = range(tr_size)
        random.seed(5)
        random.shuffle(indices)
        train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
        review = train_data[0]
        classes = train_data[1]
        features = np.array(get_feature_vectors(review, self.bin_feats))
        for num in range(self.num_iter):
            gradient = np.zeros(self.vocab_size)
            bias_gradient = 0
            for i in range(len(features)):
                if (classes[i] * (np.dot(features[i], self.weights) + self.bias)) <= 1:
                    gradient += classes[i] * features[i]
                    bias_gradient += classes[i]
                # self.weights[i+1] = self.weights[i] + self.rate * classes[i] * features[i]
                # self.bias = self.bias + self.rate * classes[i]
            self.weights += self.rate * gradient
            bias_gradient += self.rate * bias_gradient
        # raise NotImplementedError

    def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs
        input = np.array(get_feature_vectors(test_x))
        output = []
        for vec in range(len(input)):
            prediction = np.sign(np.dot(self.weights, input[vec]) + self.bias)
            if prediction == 0:
                prediction = -1
            output.append(prediction)
        return output
        #raise NotImplementedError

class SGDLog(BinaryClassifier):
    
    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.weights = np.zeros(args.vocab_size)
        self.bias = 0
        self.updates = 0
        self.rate = args.lr_sgd
        self.num_iter = args.num_iter
        self.vocab_size = args.vocab_size
        self.bin_feats = args.bin_feats
        # raise NotImplementedError
        
    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        tr_size = len(train_data[0])
        indices = range(tr_size)
        random.seed(5)
        random.shuffle(indices)
        train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
        review = train_data[0]
        classes = train_data[1]
        features = np.array(get_feature_vectors(review, self.bin_feats))
        for num in range(self.num_iter):
            gradient = np.zeros(self.vocab_size)
            bias_gradient = 0
            for i in range(len(features)):
                gradient += (classes[i] - self.sigmoid(np.dot(self.weights, features[i]))) * features[i]
                bias_gradient += classes[i] - self.sigmoid(np.dot(self.weights, features[i]))
                # self.weights[i+1] = self.weights[i] + self.rate * classes[i] * features[i]
                # self.bias = self.bias + self.rate * classes[i]
            self.weights += self.rate * gradient
            bias_gradient += self.rate * bias_gradient
        # raise NotImplementedError

    def sigmoid(self, X, derivative=False):
        if X > 700:
            return 1.0
        elif X < -700:
            return 0.0
        sigm = 1. / (1. + np.exp(-X))
        if derivative:
            return sigm * (1. - sigm)
        return sigm

    def predict(self, test_x):
        # TO DO: Compute and return the output for the given test inputs
        input = np.array(get_feature_vectors(test_x))
        output = []
        for vec in range(len(input)):
            prediction = np.sign(np.dot(self.weights, input[vec]) + self.bias)
            if prediction <= 0:
                prediction = -1
            else:
                prediction = 1
            output.append(prediction)
        return output
        # raise NotImplementedError

class SGDHingeReg(BinaryClassifier):

    def __init__(self, args):
        # TO DO: Initialize parameters here
        self.weights = np.zeros(args.vocab_size)
        self.bias = 0
        self.updates = 0
        self.rate = args.lr_sgd
        self.num_iter = args.num_iter
        self.vocab_size = args.vocab_size
        self.bin_feats = args.bin_feats
        self.la = args.la
        # raise NotImplementedError

    def fit(self, train_data):
        # TO DO: Learn the parameters from the training data
        tr_size = len(train_data[0])
        indices = range(tr_size)
        random.seed(5)
        random.shuffle(indices)
        train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
        review = train_data[0]
        classes = train_data[1]
        features = np.array(get_feature_vectors(review, self.bin_feats))
        for num in range(self.num_iter):
            gradient = np.zeros(self.vocab_size)
            bias_gradient = 0
            for i in range(len(features)):
                if (classes[i] * (np.dot(features[i], self.weights) + self.bias)) <= 1:
                    gradient += classes[i] * features[i]
                    bias_gradient += classes[i]
                # self.weights[i+1] = self.weights[i] + self.rate * classes[i] * features[i]
                # self.bias = self.bias + self.rate * classes[i]
            gradient -= self.la * self.weights
            self.weights += self.rate * gradient
            bias_gradient += self.rate * bias_gradient
        # raise NotImplementedError

    def predict(self, test_x):
        # TO DO: Compute and return the output for the given test inputs
        input = np.array(get_feature_vectors(test_x))
        output = []
        for vec in range(len(input)):
            prediction = np.sign(np.dot(self.weights, input[vec]) + self.bias)
            if prediction == 0:
                prediction = -1
            output.append(prediction)
        return output
        # raise NotImplementedError

class SGDLogReg(BinaryClassifier):

    def __init__(self, args):
        # TO DO: Initialize parameters here
        self.weights = np.zeros(args.vocab_size)
        self.bias = 0
        self.updates = 0
        self.rate = args.lr_sgd
        self.num_iter = args.num_iter
        self.vocab_size = args.vocab_size
        self.bin_feats = args.bin_feats
        self.la = args.la
        # raise NotImplementedError

    def fit(self, train_data):
        # TO DO: Learn the parameters from the training data
        tr_size = len(train_data[0])
        indices = range(tr_size)
        random.seed(5)
        random.shuffle(indices)
        train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
        review = train_data[0]
        classes = train_data[1]
        features = np.array(get_feature_vectors(review, self.bin_feats))
        for num in range(self.num_iter):
            gradient = np.zeros(self.vocab_size)
            bias_gradient = 0
            for i in range(len(features)):
                gradient += classes[i] * features[i]
                bias_gradient += classes[i]
            # self.weights[i+1] = self.weights[i] + self.rate * classes[i] * features[i]
            # self.bias = self.bias + self.rate * classes[i]
            gradient -= self.la * self.weights
            self.weights += self.rate * gradient
            bias_gradient += self.rate * bias_gradient
        # raise NotImplementedError

    def predict(self, test_x):
        # TO DO: Compute and return the output for the given test inputs
        input = np.array(get_feature_vectors(test_x))
        output = []
        for vec in range(len(input)):
            prediction = np.sign(np.dot(self.weights, input[vec]) + self.bias)
            if prediction <= 0:
                prediction = -1
            else:
                prediction = 1
            output.append(prediction)
        return output
        # raise NotImplementedError

