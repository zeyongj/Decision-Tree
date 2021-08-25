from math import log
import numpy as np
import pandas as pd
import random
from enum import Enum
import copy

# The codes are inspired and modified from the website: https://github.com/AtenrevCode/DecisionTreeClassifier.

# The following codes are about pre-processing

print('......Reading Data......\n')
training_file = "data/adult.data.csv"
testing_file = "data/adult.test.csv"
print('......Reading Completed......\n')


def readData(path):
    return pd.read_csv(path, header=0, delimiter=",")


def preprocessing():
    training_dataset = readData(training_file)
    testing_dataset = readData(testing_file)

    # Ignore fulwgt, as it does not affect classification
    training_dataset.drop("fnlwgt", axis="columns", inplace=True)

    # Ignore variables that are redundant
    training_dataset.drop("education-num", axis="columns", inplace=True)
    training_dataset.drop("capital-gain", axis="columns", inplace=True)
    training_dataset.drop("capital-loss", axis="columns", inplace=True)

    # Ignore fulwgt, as it does not affect classification
    testing_dataset.drop("fnlwgt", axis="columns", inplace=True)

    # Ignore variables that are redundant
    testing_dataset.drop("education-num", axis="columns", inplace=True)
    testing_dataset.drop("capital-gain", axis="columns", inplace=True)
    testing_dataset.drop("capital-loss", axis="columns", inplace=True)

    x_train = training_dataset.drop("income", axis="columns").to_numpy()
    y_train = training_dataset["income"].to_numpy()
    x_test = testing_dataset.to_numpy()
    return training_dataset.columns, x_train, y_train, x_test


# The following codes are about utility.
# This function calculates the entropy of the set s.
def entropy(s):
    _, counts = np.unique(s, return_counts=True)
    probability = counts / len(s)
    return -sum([prob * log(prob, 2) for prob in probability])


# This function calculates the entropy of the set s conditioned to the attribute a.
def entropy_cond(s, a):
    n_s = len(s)
    labels = np.unique(a)
    entrp = 0

    for label in labels:
        index = np.where(a == label)
        sv = s[index]
        n_sv = len(sv)
        entrp += (n_sv / n_s) * entropy(sv)

    return entrp


# Gain obtained when ramifying using the a attribute.
def gain(s, a):
    non_nan_values = np.where(a != '?')
    a_modified = a[non_nan_values]
    s_modified = s[non_nan_values]

    entrp = entropy(s)
    entrp_cond = entropy_cond(s_modified, a_modified)
    return len(s_modified) / len(s) * (entrp - entrp_cond)



# The following codes are about validation
class Measure(Enum):
    ACC = 1
    PR = 2
    F1 = 3
    REC = 4
    SPEC = 5

class Validation:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.negative, self.positive = np.unique(self.y_train)
        self.tp = self.fp = self.tn = self.fn = 0
        self.empty_measures = {Measure.ACC: [], Measure.PR: [], Measure.REC: [], Measure.SPEC: [], Measure.F1: []}

    def count_results(self, predictions, test):
        self.tp = self.fp = self.tn = self.fn = 0

        for k in range(predictions.shape[0]):
            if predictions[k] == self.positive:
                if test[k] == self.positive:
                    self.tp += 1
                else:
                    self.fp += 1
            else:
                if test[k] == self.negative:
                    self.tn += 1
                else:
                    self.fn += 1

    # This function is for testing, given that there is an attribute called test, I changed the function signature.
    def calculate_measures(self, measures, predictions, test):
        self.count_results(predictions, test)

        measures[Measure.ACC].append(self.accuracy())
        measures[Measure.PR].append(self.precision())
        measures[Measure.REC].append(self.recall())
        measures[Measure.SPEC].append(self.specificity())
        measures[Measure.F1].append(self.f1_score())


    def accuracy(self):
        return (self.tp + self.tn)/(self.tp + self.tn + self.fp + self.fn)

    def precision(self):
        return self.tp/(self.tp + self.fp)


    def recall(self):
        return self.tp/(self.tp + self.fn)


    def specificity(self):
        return self.tn/(self.fp + self.tn)


    def f1_score(self):
        precision = self.precision()
        recall = self.recall()

        return precision*recall/(precision + recall)


    def generate_folds(self, k):
        x_folds = []
        y_folds = []
        index = np.array(range(self.x_train.shape[0]))
        np.random.seed(1)
        np.random.shuffle(index)
        folds_index = np.array_split(index, k)

        for fold in folds_index:
            x_folds.append(self.x_train[fold])
            y_folds.append(self.y_train[fold])

        return x_folds, y_folds

    #    Given an integer k and a measure, it applies the k-fold validation to calculate the mean of the measures
    #    obtained with each fold.

    def score_cross_val(self, k, model):
        x_folds, y_folds = self.generate_folds(k)
        measures = self.empty_measures
        pruned_measures = self.empty_measures

        for fold in range(k):
            x_val = x_folds[fold]
            y_val = y_folds[fold]
            x_train = np.vstack(tuple(x_folds[i] for i in range(k) if i != fold))
            y_train = np.hstack(tuple(y_folds[i] for i in range(k) if i != fold))

            model.fit(x_train, y_train)
            predictions = model.predict(x_val)

            pruned_model = copy.copy(model)
            pruned_model.prune()
            pruned_model.fit(x_train, y_train)
            pruned_predictions = pruned_model.predict(x_val)

            self.calculate_measures(measures, predictions, y_val)
            self.calculate_measures(pruned_measures, pruned_predictions, y_val)
            if np.array(pruned_measures[Measure.ACC]).mean()>np.array(measures[Measure.ACC]).mean():
                model = pruned_model

            model.fit(x_train, y_train)
            predictions = model.predict(x_val)
            self.calculate_measures(measures, predictions, y_val)

        return model,measures


    # def error_reduction_pruning(self, measures, model, predictions, pruned_measures, x_train, x_val, y_train, y_val):
    #     pruned_model = copy.copy(model)
    #     pruned_model.prune()
    #     pruned_model.fit(x_train, y_train)
    #     pruned_predictions = pruned_model.predict(x_val)
    #     self.test_accuracy(measures, predictions, y_val)
    #     self.test_accuracy(pruned_predictions, pruned_predictions, y_val)
    #     if np.array(pruned_measures[Measure.ACC]).mean()>np.array(measures[Measure.ACC]).mean():
    #             model = pruned_model
    #     return model

    # It trains a model trained with all the training set and calculates the given measure for the predictions made
    # with the test set. Afterwards, it trains the final model (with all the data) and prints the resulting decision
    # tree.

    def final_measure(self, model):
        model.fit(self.x_train, self.y_train)
        predictions = model.predict(self.x_test)
        measures = self.empty_measures
        self.calculate_measures(measures, predictions, self.y_test)

        return measures

 # The following codes are about decision tree.

class Criterion(Enum):
        ID3 = 1


class DecisionTreeClassifier:

        def __init__(self, attr_headers, continuous_attr_header, criterion):
            self.criterion = criterion
            self.attr_headers = attr_headers
            self.attribute_values = []
            self.continuous_attr_header = continuous_attr_header

        def set_attribute_values(self, data):
            self.attribute_values = [np.unique(data[:, i]) for i in range(data.shape[1])]

        def set_labels(self, labels):
            self.attribute_values = labels

        # This is the grow function, given that it is called fit in the original file, I did not risk of refractoring.
        def fit(self, X, Y):
            self.model = SubTree(Y, X, self.attr_headers, self.criterion, self.attribute_values,
                                 self.continuous_attr_header)

        def predict(self, X):
            return self.model.predict(X)

        def __str__(self):
            return str(self.model)

        def prune(self):
            self.model.prune()

class SubTree:
    def __init__(self, s, A, A_header, criterion, labels, continuous_attributes_header):
        self.s = s
        self.A = A
        self.A_header = A_header
        self.negative, self.positive = np.unique(self.s)
        self.criterion = criterion
        self.continuous_attributes_header = continuous_attributes_header
        self.labels = labels
        self.continuous_partition = -1
        self.discretized_continuous_column = None
        self.attribute = self.select_attribute()
        self.child_nodes = self.develop_child_nodes()

    def prune(self):
        random_leaf = random.randint(1, len(self.child_nodes) - 1)
        dictionary_key = list(self.child_nodes.keys())[random_leaf]
        self.child_nodes.pop(dictionary_key)

    def calculate_gain(self, a):
        if self.criterion == Criterion.ID3:
            return gain(self.s, a)
        else:  # GINI
            return -1

    def attribute_continuous(self, attribute):
        return (self.A_header[attribute] in self.continuous_attributes_header)

    #        Select the best partition for a continuous attribute.

    def select_partition(self, a, max_gain, max_index,
                         attribute):  # the gain and gini_gain functions have the nan correction implemented
        values = np.unique(a)
        index = np.where(a == '?')

        for i in values[:-1]:
            a_discrete = np.where(a > i, 'major', 'minor')
            a_discrete[index] = '?'

            current_gain = self.calculate_gain(a_discrete)

            if current_gain > max_gain:
                max_gain = current_gain
                max_index = attribute
                self.continuous_partition = i
                self.discretized_continuous_column = a_discrete

        return max_gain, max_index

    # Select the best attribute for making the decision on the subtree root (based on the selection criterion).

    def select_attribute(self):  # the gain and gini_gain functions have the nan correction implemented
        max_index = 0
        max_gain = 0

        for attribute in range(len(self.A_header)):
            a = self.A.T[attribute, :]

            if not self.attribute_continuous(attribute):
                current_gain = self.calculate_gain(a)

                if current_gain > max_gain:
                    max_gain = current_gain
                    max_index = attribute
                    self.continuous_partition = -1
            else:
                max_gain, max_index = self.select_partition(a, max_gain, max_index, attribute)

        return max_index

    # It calculates the column and the values corresponding to the current attribute.

    def treat_attribute(self):
        if not self.attribute_continuous(self.attribute):
            curr_col = self.A[:, self.attribute]
            labels = self.labels[self.attribute]
            labels = labels[np.where(labels != '?')]  # we're not considering nan values
        else:
            labels = ['major', 'minor']

            if self.continuous_partition == -1:  # if the gain is 0 (same value for all the samples)
                a = self.A.T[self.attribute, :]
                curr_col = np.where(a < 0, 'major', 'minor')
            else:
                curr_col = self.discretized_continuous_column

        return curr_col, labels

    # Find the child nodes of the current decision node (that will be leaves or other decision nodes).

    def develop_child_nodes(self):
        child_nodes = {}
        curr_col, labels = self.treat_attribute()

        if self.A.shape[1] == 1:  # If it's the last attribute to expand, then the child nodes will be leaves (
            # labeled with the majority class on each attribute value)
            self.develop_last_attribute(child_nodes, curr_col, labels)
        else:
            # If not, explore each value of the attribute
            for label in labels:
                index = np.where(curr_col == label)
                sv = self.s[index]
                svu, count = np.unique(sv, return_counts=True)

                if svu.shape[0] == 1:
                    # if all the data corresponding to this attribute value belongs to the same class, the child
                    # is a leaf
                    child_nodes[label] = svu[0]
                elif svu.shape[0] == 0:
                    # If there is no data with the corresponding value, the child is a leaf labeled with the
                    # majority class on the father
                    self.develop_child_with_no_data(child_nodes, label)
                else:
                    classes = np.unique(self.A[index, 1 - self.attribute])
                    if self.A.shape[1] == 2 and classes.shape[0] == 1:  # special case, if not, on the next
                        # generated node every leaf will have the same class
                        child_nodes[label] = svu[np.argmax(count)]
                    else:
                        if self.continuous_partition != -1:  # If the attribute is continuous and it has a
                            # positive gain, we don't delete the attribute
                            child_nodes[label] = SubTree(sv, self.A[index],
                                                         self.A_header, self.criterion,
                                                         self.labels,
                                                         self.continuous_attributes_header)
                        else:
                            child_nodes[label] = SubTree(sv, np.delete(self.A[index], self.attribute, 1),
                                                         np.delete(self.A_header, self.attribute), self.criterion,
                                                         [self.labels[i] for i in range(len(self.labels)) if
                                                          i != self.attribute],
                                                         self.continuous_attributes_header)

        return child_nodes

    # If it's the last attribute to expand, then the child nodes will be leaves (labeled with the majority class on
    # each attribute value)

    def develop_last_attribute(self, child_nodes, curr_col, labels):
        for label in labels:
            index = np.where(curr_col == label)
            sv = self.s[index]
            classes, count = np.unique(sv, return_counts=True)

            if len(classes) == 0:  # No data
                self.develop_child_with_no_data(child_nodes, label)
            elif len(classes) != 1 and len(np.unique(count)) == 1:  # improbable case
                child_nodes[label] = classes[0]
            else:
                child_nodes[label] = classes[np.argmax(count)]

    #    If there is no data to the corresponding attribute value, it will be assigned the majority class on the father.

    def develop_child_with_no_data(self, child_nodes, label):
        classes_father, count_father = np.unique(self.s, return_counts=True)

        if len(classes_father) != 1 and len(np.unique(count_father)) == 1:  # improbable case
            child_nodes[label] = classes_father[0]  # '?'
        else:
            child_nodes[label] = classes_father[np.argmax(count_father)]

    #    It converts the values of a continuous attribute to the values 'major' or 'menor' depending on the partition
    #    of the attribute.

    def convert_continuous_attribute(self, label):
        if self.continuous_partition == -1:  # if the gain was 0, then every sample has the same value
            continuous_partition = self.A.T[self.attribute, :][0]
        else:
            continuous_partition = self.continuous_partition

        if label > continuous_partition:
            label = 'major'
        else:
            label = 'minor'

        return label

    #    It counts the number of classes on a leaf of the tree, depending if it's the first iteration or not of the
    #    predict_single_count function.

    def count_classes(self, first_iteration, child_node, label, n):
        if first_iteration:
            curr_col = self.A[:, self.attribute]
            count = len(np.where(curr_col == label)[0])
        else:
            count = n

        if child_node == self.negative:
            return count, 0
        else:
            return 0, count

    # Returns the number of samples belonging to each class  on de decision nodes and leaves given by the set of
    # attributes X. If it's the first time that this function is called (i.e. the current attribute is actually NaN),
    # only the cases with the considered value of the current attribute will be counted. If it's called recursively,
    # every sample on the final decision node will be counted.

    def predict_single_count(self, X, first_iteration):
        label = X[self.attribute]
        n = self.s.shape[0]

        if label == '?':
            neg, pos = self.predict_nan_value(X)
            return n * neg, n * pos
        else:
            if self.attribute_continuous(self.attribute):
                label = self.convert_continuous_attribute(label)

            child_node = self.child_nodes[label]

            if type(child_node) is SubTree:
                if self.continuous_partition == -1:
                    return child_node.predict_single_count(np.delete(X, self.attribute), False)
                else:
                    return child_node.predict_single_count(X, False)
            else:
                return self.count_classes(first_iteration, child_node, label, n)

    #   Returns the probabilities of belonging to the positive or negative class for the set of attributes X,
    #   where the current attribute is NaN.

    def predict_nan_value(self, X):
        prob_positive = 0
        prob_negative = 0
        labels = self.labels[self.attribute]
        labels = labels[np.where(labels != '?')]
        n = self.s.shape[0]

        for label in labels:
            X[self.attribute] = label
            neg, pos = self.predict_single_count(X, True)
            prob_positive += pos / n
            prob_negative += neg / n

        return prob_negative, prob_positive

#    Returns the prediction for the set of attributes X.

    def predict_single(self, X):
        label = X[self.attribute]

        if label == '?':
            prob_negative, prob_positive = self.predict_nan_value(X)

            if prob_positive > prob_negative:
                return self.positive
            else:
                return self.negative
        else:

            if self.attribute_continuous(self.attribute):
                label = self.convert_continuous_attribute(label)

            child_node = self.child_nodes[label]

            if type(child_node) is SubTree:
                if self.continuous_partition == -1:
                    return child_node.predict_single(np.delete(X, self.attribute))
                else:
                    return child_node.predict_single(X)
            else:
                return child_node

#    Returns the prediction for each set of attributes on the list X.

    def predict(self, X):
        if len(X.shape) == 1:
            return self.predict_single(X)
        else:
            return np.array([self.predict_single(x) for x in X])

    def __str__(self, level=0):
        tabs = level * '\t'
        output = f'Attribute {self.A_header[self.attribute]}:\n'

        i = ''
        if self.attribute_continuous(self.attribute):
            if self.continuous_partition == -1:  # if the gain was 0, then every sample has the same value
                i = ' that ' + str(self.A.T[self.attribute, :][0])
            else:
                i = ' that ' + str(self.continuous_partition)

        for label, child_node in self.child_nodes.items():
            if type(child_node) is SubTree:
                output += '\t' + tabs + 'Value=' + str(label) + i + ': ' + child_node.__str__(level + 1)
            else:
                output += '\t' + tabs + 'Value=' + str(label) + i + ': class = ' + str(child_node) + '\n'

        return output

    def select_attribute_discrete(self):  # the gain and gini_gain functions have implemented the nan correction
        if self.criterion == Criterion.ID3:
            gain_attributes = np.array([gain(self.s, a) for a in self.A.T])

        return np.argmax(gain_attributes)

    def develop_child_nodes_discrete(self):
        child_nodes = {}
        curr_col = self.A[:, self.attribute]
        labels = self.labels[self.attribute]
        labels = labels[np.where(labels != '?')]  # we're not considering nan values

        if self.A.shape[1] == 1:
            self.develop_last_attribute(child_nodes, curr_col, labels)

        else:
            # If not, explore each value of the attribute
            for label in labels:
                index = np.where(curr_col == label)
                sv = self.s[index]
                svu = np.unique(sv)

                if len(svu) == 1:
                    # if all the data corresponding to this attribute value belongs to the same class, the child is a
                    # leaf
                    child_nodes[label] = svu[0]
                elif len(svu) == 0:
                    # If there is no data with the corresponding value, the child is a leaf labeled with the majority
                    # class on the father
                    self.develop_child_with_no_data(child_nodes, label)
                else:
                    child_nodes[label] = SubTree(sv, np.delete(self.A[index], self.attribute, 1),
                                                 np.delete(self.A_header, self.attribute), self.criterion,
                                                 [self.labels[i] for i in range(len(self.labels)) if
                                                  i != self.attribute],
                                                 self.continuous_attributes_header)

        return child_nodes

# To leave a certain percent of the training data as validation data
def split_validation(x_train, y_train, split_ratio):
    length = len(x_train)
    my_list = list(range(length))
    split_size = length * split_ratio
    value_index = random.sample(my_list, int(split_size))
    value_of_x = x_train[value_index]
    value_of_y = y_train[value_index]
    for index in value_index:
        my_list.remove(index)
    modified_x_train = x_train[my_list]
    modified_y_train = y_train[my_list]
    return modified_x_train, modified_y_train, value_of_x, value_of_y



def main():
    columns, x_train, y_train, x_test = preprocessing()
    modified_x_train, modified_y_train, value_of_x, value_of_y = split_validation(x_train, y_train, 0.1)

    decision_tree = DecisionTreeClassifier(columns[:-1], ['age', 'hours-per-week', 'capital-gain', 'capital-loss'],
                                                             Criterion.ID3)

    decision_tree.set_attribute_values(np.vstack((modified_x_train, value_of_x)))
    validation = Validation(x_train, y_train, value_of_x, value_of_y)

    print('......Processing 5-fold Validation......\n')
    decision_tree, score = validation.score_cross_val(5, decision_tree)
    print(f'Mean Accuracy of 5-fold Validation Is: {np.array(score[Measure.ACC]).mean()}\n')
    print('......5-fold Validation Completed......\n')

    print('......Generating Final Decision Tree......\n')
    final_measure = validation.final_measure(decision_tree)
    print(f'Accuracy of Final Decision Tree Is: {np.array(final_measure[Measure.ACC]).mean()}\n')
    print('......Final Decision Tree Generated......\n')

    decision_tree.fit(x_train, y_train)
    y_test = decision_tree.predict(x_test)
    df = pd.DataFrame(y_test)
    print('......Creating Prediction File......\n')
    df.to_csv("predictions.csv", index = False, sep = ',')
    print('......Prediction File Created......\n')


if __name__ == "__main__":
    main()
