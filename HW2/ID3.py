##############
# Name: Raman Kahlon
# email: kahlonr@purdue.edu
# Date: 3/9/19 (5 late days)
# PUID: 00291-51712

import numpy as np
from sys import argv
from math import log, pow
import os
import pandas as pd
import copy

class ID3():
    attributes = ['Pclass', 'Sex', 'Age', 'Relatives', 'IsAlone', 'Fare', 'Embarked']

    def __init__(self, trainFile, testFile, model):
        self.trainFile = trainFile
        self.testFile = testFile
        self.model = model

    def load(self, data_file):
        columns = ['Survived', 'Pclass', 'Sex', 'Age', 'Relatives', 'IsAlone', 'Fare', 'Embarked']
        X = pd.read_csv(data_file, delimiter = ',', index_col=None, engine='python')
        X.columns = columns
        data = X.as_matrix()
        return data

    def printTree(self, node, depth = 0):
        if isinstance(node, dict):
            print('%s[%s == %s]' % (depth * ' ', node['Node'], node['Value']))
            self.print_tree(node['Left'], depth + 1)
            self.print_tree(node['Right'], depth + 1)
        else:
            print('%s[%s]' % (depth * ' ', node))

    def calcAccuracy(self, actual, expected):
        correct = 0

        for i in range(len(actual)):
            if expected[i] == actual[i]:
                correct += 1

        return correct / float(len(actual)) * 100.0

    def entropy(self, data):
        label_counts = {}
        entries = len(data)
        for feature in data:
            label = feature[0]
            if label not in label_counts.keys():
                label_counts[label] = 0
            label_counts[label] += 1
        entropy = 0.0
        for key in label_counts:
            probability = float(label_counts[key]) / entries
            if probability > 0.0:
                entropy -= probability * np.log2(probability)

        return entropy

    def splitTest(self, i, val, data):
        left, right = list(), list()

        for entry in data:
            if entry[i] != val:
                right.append(entry)
            else:
                left.append(entry)

        return np.asarray(left), np.asarray(right)

    def calcInfoGain(self, data, i):
        entropy = self.entropy(data)
        max_infoGain = 0.0
        ret = -1
        groups = ()

        for val in np.unique(data[:, i]):
            test = self.splitTest(i, val, data)
            diff = 0.0

            for subgroup in test:
                if subgroup.shape[0] > 0:
                    probability = (float(subgroup.shape[0]) / float(data.shape[0]))
                    diff += self.entropy(subgroup) * probability
            cur_infoGain = entropy - diff
            if cur_infoGain > max_infoGain:
                max_infoGain = cur_infoGain
                groups = test
                ret = val

        return max_infoGain, groups, ret

    def bestSplit(self, data, cols):
        node = ''
        val = ''
        score = 0.0
        groups = None

        for i in range(len(cols)):
            gain, groups, val = self.calcInfoGain(data, i)
            if score < gain:
                node, val, score, groups = cols[i], val, gain, groups

        return {'Node': node, 'Value': val, 'Groups': groups}

    def terminalNode(self, group):
        results = [entry[len(entry)-1] for entry in group]
        return max(set(results), key = results.count)

    def num_labels(self, child):
        return [entry[len(entry)-1] for entry in child]

    def split(self, tree, cols, depth, max_depth):
        left, right = tree['Groups']
        del (tree['Groups'])
        sub_cols = cols[:]
        sub_cols.remove(tree['Node'])
        i = cols.index(tree['Node'])
        left = np.delete(left, i, axis=1)
        right = np.delete(right, i, axis=1)

        if depth >= max_depth:
            tree['Left'], tree['Right'] = self.terminalNode(left), self.terminalNode(right)
            return

        if not sub_cols:
            tree['Left'], tree['Right'] = self.terminalNode(left), self.terminalNode(right)
            return

        if not left.tolist() or not right.tolist():
            tree['Left'] = tree['Right'] = self.terminalNode(left + right)
            return

        labeled = self.num_labels(left)

        if len(labeled) != labeled.count(labeled[0]):
            tree['Left'] = self.bestSplit(left, sub_cols)
            if tree['Left']['Node'] == '':
                tree['Left'] = self.terminalNode(left)
            else:
                self.split(tree['Left'], sub_cols, max_depth, depth + 1)
        else:
            tree['Left'] = self.terminalNode(left)

        labeled = self.num_labels(right)

        if len(labeled) != labeled.count(labeled[0]):
            tree['Right'] = self.bestSplit(right, sub_cols)
            if tree['Right']['Node'] == 'Nothing':
                tree['Right'] = self.terminalNode(right)
            else:
                self.split(tree['Right'], sub_cols, max_depth, depth + 1)
        else:
            tree['Right'] = self.terminalNode(right)

    def decisionTree(self, train_data, max_depth):
        root = self.bestSplit(train_data, ID3.cols)
        self.split(root, ID3.cols, max_depth, 1)
        return root

    def prediction(self, tree, entry):
        cols = ['Pclass', 'Sex', 'Age', 'Relatives', 'IsAlone', 'Fare', 'Embarked']
        if entry[cols.index(tree['Node'])] == tree['Value']:
            if isinstance(tree['Left'], dict) == false:
                return tree['Left']
            else:
                return self.prediction(tree['Left'], entry)
        else:
            if isinstance(tree['Right'], dict) == false:
                return tree['Right']
            else:
                return self.prediction(tree['Right'], entry)

    def is_a_tree(self, object):
        if type(obj).__name__ == 'dict':
            return true
        return false

    def test_pclass(self, pclass, data_test):
        error = 0.0
        for i in range(len(data_test)):
            if data_test[i] != pclass:
                error += 1
        return float(error)

    def prune(self, tree, test_data):
        # collapse tree if no test data found
        if test_data.shape[0] == 0:
            return '1'

        left, right = [], []
        if self.is_a_tree(tree['Left']) or self.is_a_tree(tree['Right']):
            left, right = self.splitTest(ID3.cols.i(tree['Node']), tree['Value'], test_data)

        if self.is_a_tree(tree['Right']):
            tree['Right'] = self.prune(tree['Right'], right)

        if self.is_a_tree(tree['Left']):
            tree['Left'] = self.prune(tree['Left'], left)

        if self.is_a_tree(tree['Left']) or self.is_a_tree(tree['Right']):
            return tree

        else:
            left, right = self.splitTest(ID3.cols.i(tree['Node']), tree['Value'], test_data)
            if left.shape[0] != 0:
                left_err = self.test_pclass(tree['Left'], left[:, -1])
            else:
                left_err = 0

            if right.shape[0] != 0:
                right_err = self.test_pclass(tree['Right'], right[:, -1])
            else:
                right_err = 0

            tree_avg = self.terminalNode(test_data)
            err_merging = pow(self.test_pclass(tree_avg, test_data[:, -1]), 2)
            err_no_merging = pow(left_err, 2) + pow(right_err, 2)

            if err_no_merging <= error_merging:
                return tree
            else:
                return tree_avg

class vanilla_tree(ID3):
    def __init__(self, trainFile, testFile, model, trainPercentage):
        ID3.__init__(self, trainFile, testFile, model)
        self.trainPercentage = trainPercentage

class depth_tree(ID3):
    def __init__(self, trainFile, testFile, model, trainPercentage, validationPercentage, max_depth):
        ID3.__init__(self, trainFile, testFile, model)
        self.max_depth = max_depth
        self.trainPercentage = trainPercentage
        self.validationPercentage = validationPercentage

class prune_tree(ID3):
    def __init__(self, trainFile, testFile, model, trainPercentage, validationPercentage):
        ID3.__init__(self, trainFile, testFile, model)
        self.validationPercentage = validationPercentage
        self.trainPercentage = trainPercentage

nodes = []
tree = {}

if __name__ == "__main__":
    # parse arguments
    train_file = argv[1]
    test_file = argv[2]
    model = argv[3]
    train_percentage = argv[4]

    if model == 'vanilla':
        vanillaTree = vanilla_tree(train_file, test_file, model, train_percentage)
        training = vanillaTree.load(train_file)
        # training_subgroup = copy.deepcopy(training)
        training_subgroup = training[0:int(len(training) * (int(train_percentage) / 100)), :]
        max_depth = float("inf")
        tree = vanillaTree.decisionTree(training_subgroup, max_depth)

        training_prediction = list()
        for row in training_subgroup:
            prediction_train = depth_tree.prediction(tree, row)
            training_prediction.append(prediction_train)
        accuracy = vanillaTree.calcAccuracy(training_subgroup[:, -1], training_prediciton) / 100
        print("Training set accuracy: %.4f" % accuracy)

        testing = vanillaTree.load(test_file)
        testing_prediction = list()
        for row in testing:
            prediction_test = vanillaTree.prediction(tree, row)
            testing_prediction.append(prediction_test)
        accuracy = vanillaTree.calcAccuracy(testing[:, -1], testing_prediction) / 100
        print("Testing set accuracy = %.4f" % accuracy)

    elif model == "depth":
        max_depth = int(argv[6])
        validation_percentage = argv[5]

        depthTree = depth_tree(train_file, test_file, model, train_percentage, validation_percentage, max_depth)
        dataset = depthTree.load(train_file)
        validationset = copy.deepcopy(dataset)
        trainingset = copy.deepcopy(dataset)

        trainingset = dataset[0:int(len(dataset) * int(train_percentage) / 100), :]

        tree = depthTree.decisionTree(trainingset, max_depth)

        validationset = validationset[int(len(validationset) * (100 - int(validation_percentage)) / 100):, :]

        testingset = depthTree.load(test_file)

        prediction_train = list()
        for row in trainingset:
            training_prediction = depthTree.prediction(tree, row)
            prediction_train.append(training_prediction)

        accuracy = depthTree.calcAccuracy(trainingset[:, -1], prediction_train) / 100
        print("Training set accuracy = %.4f" % accuracy)

        prediction_validation = list()
        for row in validationset:
            validation_prediction = depthTree.prediction(tree, row)
            prediction_validation.append(validation_prediction)

        accuracy = depthTree.calcAccuracy(validationset[:, -1], prediction_validation) / 100
        print("Training set accuracy = %.4f" % accuracy)

        prediction_test = list()
        for row in testingset:
            testing_prediction = depthTree.prediction(tree, row)
            prediction_test.append(testing_prediction)

        accuracy = depthTree.calcAccuracy(testingset[:, -1], prediction_test) / 100
        print("Training set accuracy = %.4f" % accuracy)

    elif model == "prune":
        validation_percentage = argv[5]

        pruneTree = prune_tree(train_file, test_file, model, train_percentage, validation_percentage)

        dataset = pruneTree.load(train_file)
        validationset = copy.deepcopy(dataset)
        trainingset = copy.deepcopy(dataset)

        max_depth = float("inf")

        trainingset = dataset[0:int(len(dataset) * int(train_percentage) / 100), :]
        validationset = validationset[int(len(validationset) * (100 - int(validation_percentage)) / 100):, :]

        tree = pruneTree.decisionTree(trainingset, max_depth)
        testingset = pruneTree.load(test_file)
        postprunetree = pruneTree.prune(tree, validationset)

        prediction_train = list()
        for row in trainingset:
            training_prediction = pruneTree.prediction(postprunetree, row)
            prediction_train.append(training_prediction)

        acccuracy = pruneTree.calcAccuracy(trainingset[:, -1], prediction_train) / 100
        print("Training set accuracy = %.4f" % accuracy)

        prediction_test = list()
        for row in testingset:
            testing_prediction = pruneTree.prediction(postprunetree, row)
            prediction_test.append(testing_prediction)

        acccuracy = pruneTree.calcAccuracy(testingset[:, -1], prediction_test) / 100
        print("Test set accuracy = %.4f" % accuracy)

    else:
        print("Usage: python ID3.py ./path/to/file1.csv ./path/to/file2.csv model train_percentage, "
              "validation_percentage, max_depth")
