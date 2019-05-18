#!/usr/bin/env python
# coding: utf-8
# Raman Kahlon
# PUID: 00291-51712
# Submission Date: 4/4/19

from classifier import BinaryClassifier
import numpy as np
from utils import get_feature_vectors

class NaiveBayes(BinaryClassifier):
    
    def __init__(self, args):
        #TO DO: Initialize parameters here
        self.vocab_size = args.vocab_size
        self.bin_feats = args.bin_feats
        self.pos = dict()
        self.neg = dict()
        #raise NotImplementedError
        
    def fit(self, train_data):
        #TO DO: Learn the parameters from the training data
        review = train_data[0]
        classes = train_data[1]
        positive = np.array(len(review))
        pos_count = 0
        negative = np.array(len(review))
        neg_count = 0
        for i in range(len(review)):
            if classes[i] == 1:
                positive[pos_count] = classes[i]
                pos_count += 1
            else:
                negative[neg_count] = classes[i]
                neg_count -= 1
        prob_pos = pos_count / (pos_count + neg_count)
        prob_neg = neg_count / (pos_count + neg_count)
        pos_cond = 0
        neg_cond = 0
        for i in range(len(positive)):
            for j in range(len(positive[i])):
                if positive[i][j] in d.keys():
                    self.pos[positive[i][j]] += 1
                else:
                    self.pos[positive[i][j]] = 1
                # d[review[i][j]] += 1
        for i in range(len(negative)):
            for j in range(len(negative[i])):
                if negative[i][j] in d.keys():
                    self.neg[negative[i][j]] += 1
                else:
                    self.neg[negative[i][j]] = 1
                # d[review[i][j]] += 1
        for key in self.pos:
            self.pos[key] = np.log(float((self.pos[key] + 1)) / (len(positive) + self.vocab_size))
        for key in self.neg:
            self.neg[key] = np.log(float((self.neg[key] + 1)) / (len(negative) + self.vocab_size))
        #raise NotImplementedError
        
    def predict(self, test_x):
        #TO DO: Compute and return the output for the given test inputs
        # Compare with features
        score = 0
        output = []
        for i in range(len(test_x)):
            for j in range(len(test_x[i])):
                if self.pos[test_x[i][j]] > self.neg[test_x[i][j]]:
                    score += 1
                elif self.pos[test_x[i][j]] <= self.neg[test_x[i][j]]:
                    score -= 1
            if score > 0:
                output.append(1)
            else:
                output.append(-1)
            score = 0
        return output
        # raise NotImplementedError

