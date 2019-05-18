#!/usr/bin/env python
# coding: utf-8
# Raman Kahlon
# PUID: 00291-51712
# Submission Date: 4/4/19

from classifier import BinaryClassifier
import numpy as np
from utils import get_feature_vectors
import random

class Perceptron(BinaryClassifier):

    def __init__(self, args):
        #TO DO: Initialize parameters here

        # initialize weight vector, bias, learning rate
        # loop through each input vector and adjust weight vector, bias using learning rule
        # if error is 0, continue to next input vector
        # if error not 0, update the weight vector
        # self.args = args
        self.weights = np.zeros(args.vocab_size)
        self.bias = 0
        self.updates = 0
        self.rate = args.lr
        self.num_iter = args.num_iter
        # raise NotImplementedError

    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        # TO DO: Randomize the review and classes lists, but each review should match up with each class
        tr_size = len(train_data[0])
        indices = range(tr_size)
        random.seed(5)
        random.shuffle(indices)
        train_data = ([train_data[0][i] for i in indices], [train_data[1][i] for i in indices])
        review = train_data[0]
        classes = train_data[1]
        features = np.array(get_feature_vectors(review))
        for num in range(self.num_iter):
            # print "num = " + str(num)
            for i in range(len(features)):
                # print "i = " + str(i)
                a = np.dot(self.weights, features[i]) + self.bias
                prediction = np.sign(a)
                if prediction == 0:
                    prediction = -1
                if prediction != classes[i]:
                    self.weights = self.weights + self.rate * classes[i] * features[i]
                    self.bias = self.bias + self.rate * classes[i]
        return self.weights, self.bias
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
        # raise NotImplementedError


class AveragedPerceptron(BinaryClassifier):

    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.weights = np.zeros(args.vocab_size)
        self.weight_prime = 0
        self.bias = 0
        self.cached_weights = np.zeros(args.vocab_size)
        self.cached_bias = 0
        self.survival = 1
        self.updates = 0
        self.rate = args.lr
        self.num_iter = args.num_iter
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
        features = np.array(get_feature_vectors(review))
        for num in range(self.num_iter):
            for i in range(len(features)):
                # print "i = " + str(i)
                a = np.dot(self.weights, features[i]) + self.bias
                prediction = np.sign(a)
                if prediction == 0:
                    prediction = -1
                if prediction == classes[i]:
                    self.survival = self.survival + 1
                else:
                    self.weight_prime = self.weights + self.rate * classes[i] * features[i]
                    self.weights = ((self.survival * self.weights) + self.weight_prime)/(self.survival + 1)
                    self.bias = self.bias + (self.rate * classes[i])/(self.survival + 1)
                    self.survival = 1
                #     for k in range(len(self.cached_weights)):
                #         self.cached_weights[k] = self.weights[k] + self.rate * classes[i] * self.counter * features[j]
                #     self.cached_bias = self.cached_bias + self.rate * classes[i] * self.counter
                # self.counter += 1
        return self.weights, self.bias
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
        # raise NotImplementedError

