from enum import unique

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.validation import validate_data

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5: 0.45,
                 0.25: 1.32,
                 0.1: 2.71,
                 0.05: 3.84,
                 0.0001: 100000},
             2: {0.5: 1.39,
                 0.25: 2.77,
                 0.1: 4.60,
                 0.05: 5.99,
                 0.0001: 100000},
             3: {0.5: 2.37,
                 0.25: 4.11,
                 0.1: 6.25,
                 0.05: 7.82,
                 0.0001: 100000},
             4: {0.5: 3.36,
                 0.25: 5.38,
                 0.1: 7.78,
                 0.05: 9.49,
                 0.0001: 100000},
             5: {0.5: 4.35,
                 0.25: 6.63,
                 0.1: 9.24,
                 0.05: 11.07,
                 0.0001: 100000},
             6: {0.5: 5.35,
                 0.25: 7.84,
                 0.1: 10.64,
                 0.05: 12.59,
                 0.0001: 100000},
             7: {0.5: 6.35,
                 0.25: 9.04,
                 0.1: 12.01,
                 0.05: 14.07,
                 0.0001: 100000},
             8: {0.5: 7.34,
                 0.25: 10.22,
                 0.1: 13.36,
                 0.05: 15.51,
                 0.0001: 100000},
             9: {0.5: 8.34,
                 0.25: 11.39,
                 0.1: 14.68,
                 0.05: 16.92,
                 0.0001: 100000},
             10: {0.5: 9.34,
                  0.25: 12.55,
                  0.1: 15.99,
                  0.05: 18.31,
                  0.0001: 100000},
             11: {0.5: 10.34,
                  0.25: 13.7,
                  0.1: 17.27,
                  0.05: 19.68,
                  0.0001: 100000}}



def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    validate_data(data)

    data_size = data.shape[0]
    if data_size == 0:
        return 0.0

    # get the unique labels count
    _, count = np.unique(data[:, -1], return_counts=True)

    # calculate the relative sizes
    rel_sizes = count / data_size

    # calculate the gini impurity
    gini = 1 - np.sum(rel_sizes ** 2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    validate_data(data)

    data_size = data.shape[0]
    if data_size == 0:
        return 0.0

    # get the unique labels count
    _, count = np.unique(data[:, -1], return_counts=True)

    # calculate the relative sizes
    rel_sizes = count / data_size
    rel_sizes = rel_sizes * np.log2(rel_sizes)

    # calculate the entropy
    entropy = -np.sum(rel_sizes)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy


class DecisionNode:

    def __init__(self, data, impurity_func, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):

        self.data = data  # the data instances associated with the node
        self.terminal = False  # True iff node is a leaf
        self.feature = feature  # column index of feature/attribute used for splitting the node
        self.pred = self.calc_node_pred()  # the class prediction associated with the node
        self.depth = depth  # the depth of the node
        self.children = []  # the children of the node (array of DecisionNode objects)
        self.children_values = []  # the value associated with each child for the feature used for splitting the node
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.chi = chi  # the P-value cutoff used for chi square pruning
        self.impurity_func = impurity_func  # the impurity function to use for measuring goodness of a split
        self.gain_ratio = gain_ratio  # True iff GainRatio is used to score features
        self.feature_importance = 0

    def calc_node_pred(self):
        """
        Calculate the node's prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        vals, counts = np.unique(self.data[:, -1], return_counts=True)

        # if the node is pure, return the label
        if len(vals) == 1:
            pred = vals[0]
        # if the node is not pure, return the most common label
        else:
            pred = vals[0] if counts[0] >= counts[1] else vals[1]

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        if type(node) != DecisionNode:
            raise TypeError("node must be a DecisionNode")

        self.children.append(node)
        self.children_values.append(val)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting
                  according to the feature values.
        """
        goodness = 0
        groups = {}  # groups[feature_value] = data_subset
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if type(feature) != int:
            raise TypeError(f'feature must be an integer in the range of 0 and {self.data.shape[1] - 1}')
        if feature < 0 or feature >= self.data.shape[1]:
            raise IndexError('feature index out of range')

        values = np.unique(self.data[:, feature])

        # Construct subsets of data according to feature values
        groups = {val: self.data[self.data[:, feature] == val] for val in values}

        if self.gain_ratio:
            # If gain_ratio is True, use entropy to calculate information gain
            # regardless of what self.impurity_func is

            # Calculate parent entropy
            total_entropy = calc_entropy(self.data)

            # Calculate weighted sum of children entropies
            weighted_entropy_sum = 0
            for val in groups:
                weight = groups[val].shape[0] / self.data.shape[0]
                weighted_entropy_sum += weight * calc_entropy(groups[val])

            # Calculate information gain using entropy
            info_gain = total_entropy - weighted_entropy_sum

            # Calculate split information for normalization
            split_info = 0
            for val in groups:
                weight = groups[val].shape[0] / self.data.shape[0]
                if weight > 0:  # Avoid log(0)
                    split_info -= weight * np.log2(weight)

            # Calculate gain ratio
            if split_info != 0:
                goodness = info_gain / split_info
            else:
                goodness = 0
        else:
            # If gain_ratio is False, use the original impurity function
            total_impurity = self.impurity_func(self.data)

            # Calculate weighted sum of children impurities
            weighted_impurity_sum = 0
            for val in groups:
                weight = groups[val].shape[0] / self.data.shape[0]
                weighted_impurity_sum += weight * self.impurity_func(groups[val])

            # Calculate information gain using the original impurity function
            goodness = total_impurity - weighted_impurity_sum

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return goodness, groups

    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.

        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in
        self.feature_importance
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if type(n_total_sample) != int:
            raise TypeError("n_total_sample must be an integer")
        if n_total_sample <= 0:
            raise ValueError("n_total_sample must be greater than 0")

        if self.feature != -1:
            rel_size = self.data.shape[0] / n_total_sample
            self.feature_importance = rel_size * self.goodness_of_split(self.feature)[0]

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Cannot split further
        if self.terminal:
            return

        # Don't split if we've reached max depth
        if self.depth >= self.max_depth:
            self.terminal = True
            return

        # Check if the node is pure
        if len(np.unique(self.data[:, -1])) == 1:
            self.terminal = True
            return

        best_feature = -1
        max_gos = -1
        split = {}

        # Find the best feature to split on
        for feature in range(self.data.shape[1] - 1):
            if feature == self.feature:
                continue
            cur_gos, groups = self.goodness_of_split(feature)
            if cur_gos > max_gos:
                max_gos = cur_gos
                best_feature = feature
                split = groups

        # If no improvement in impurity, make this a terminal node
        if max_gos <= 0:
            self.terminal = True
            return

        if self.chi != 1:
            subsets = split.values()
            proportions = np.unique(self.data[:, -1], return_counts = True)[1] / self.data.shape[0]
            extended_chi_square = self.calc_chi_square(subsets, proportions)
            #set relevant distribution
            dof = len(proportions) - 1

            #PRUNE
            if extended_chi_square < chi_table[dof][self.chi]:
                self.terminal = True
                return

        self.feature = best_feature
        self.calc_feature_importance(self.data.shape[0])

        self.children = []
        self.children_values = []

        # Create child nodes
        for val in split:
            child = DecisionNode(
                data=split[val],
                impurity_func=self.impurity_func,
                feature=-1,  # Initialize with no feature selected
                depth=self.depth + 1,  # Increment depth
                chi=self.chi,
                max_depth=self.max_depth,
                gain_ratio=self.gain_ratio
            )
            self.add_child(child, val)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    #auxilary function for chi pruning in split
    def calc_chi_square(self, subsets, proportions):
        """
        Calculate the chi-square statistic for node pruning.

        Input:
        - subsets: Data subsets after splitting (list of numpy arrays)
        - proportions: Class proportions from parent node

        Returns:
        - extended_chi_square: Sum of chi-square statistics across all subsets
        """

        # Get unique labels in the original dataset
        unique_labels = np.unique(self.data[:, -1])
        k = len(unique_labels)
        extended_chi_square = 0

        for subset in subsets:
            if subset.shape[0] == 0:  # Skip empty subsets
                continue

            chi_square = 0
            subset_size = subset.shape[0]

            # Get counts for each class in this subset
            for i, label in enumerate(unique_labels):
                # Count occurrences of this label in the subset
                observed = np.sum(subset[:, -1] == label)
                # Expected count based on original proportions
                expected = subset_size * proportions[i]

                # Avoid division by zero
                if expected > 0:
                    chi_square += ((observed - expected) ** 2) / expected

            extended_chi_square += chi_square

        return extended_chi_square

    def validate_data(data):
        """
        Validate the input data.

        Input:
        - data: any dataset where the last column holds the labels.

        Returns:
        - None if the data is valid
        - raises ValueError if the data is invalid
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("data must be a numpy array")
        if data.ndim != 2:
            raise ValueError("data must be a 2D numpy array")
        if data.shape[1] < 2:
            raise ValueError("data must have at least two columns")

class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # the training data used to construct the tree
        self.root = None  # the root node of the tree
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.chi = chi  # the P-value cutoff used for chi square pruning
        self.impurity_func = impurity_func  # the impurity function to be used in the tree
        self.gain_ratio = gain_ratio  #
        self.height = 0 #keep track of tree height (also tree depth)

    def depth(self):
        return self.root.depth

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset.
        You are required to fully grow the tree until all leaves are pure
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        self.root = DecisionNode(data=self.data, impurity_func=self.impurity_func, feature=-1, depth=0, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)

        # Build the tree recursively
        def build_recursive(node):
            # Try to split the node
            node.split()

            # If the node is now terminal, stop recursion
            if node.terminal:
                self.height = max(self.height, node.depth)
                return

            # Otherwise, recursively build each child
            for child in node.children:
                build_recursive(child)

        # Start the recursive building process
        build_recursive(self.root)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, instance):
        """
        Predict a given instance

        Input:
        - instance: an row vector from the dataset. Note that the last element
                    of this vector is the label of the instance.

        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if not isinstance(instance, np.ndarray):
            raise TypeError("instance must be a numpy array")
        if instance.shape[1] != self.data.shape[1]:
            raise ValueError("instance must have the same number of features as the dataset")

        cur = self.root
        # go down the tree until a leaf is reached
        while not cur.terminal:
            # take the value of the instance in the splitting feature
            val = instance[cur.feature]
            if val not in cur.children_values:
                break
            # get the index of the child node corresponding to the value
            idx = cur.children_values.index(val)
            # make cur point to the corresponding child node
            cur = cur.children[idx]

        pred = cur.pred

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset

        Input:
        - dataset: the dataset on which the accuracy is evaluated

        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        validate_data(dataset)

        if dataset.shape[0] == 0:
            raise ValueError("dataset must have at least one row")

        labels = dataset[:, -1]
        n = len(labels)
        # store predictions here
        prediction = np.zeros(n, dtype=labels.dtype)

        for i in range(n):
            prediction[i] = self.predict(dataset[i])

        # calc accuracy
        accuracy = 100 * np.mean(prediction == labels)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy


def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output: the training and validation accuracies per max depth
    """
    training = []
    validation = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        validate_data(X_train)
        validate_data(X_validation)

        train_tree = DecisionTree(X_train, impurity_func=calc_entropy, max_depth=max_depth, gain_ratio=True)
        train_tree.build_tree()
        training.append(train_tree.calc_accuracy(X_train))
        validation.append(train_tree.calc_accuracy(X_validation))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return training, validation


def chi_pruning(X_train, X_test):
    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc = []
    depth = []

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    validate_data(X_train)
    validate_data(X_test)

    p_value_cutoff = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]

    for p in p_value_cutoff:
        # Create a DecisionTree object with the specified chi value
        tree = DecisionTree(X_train, impurity_func=calc_entropy, chi=p, gain_ratio=True)

        # Build the tree
        tree.build_tree()

        # Calculate training and validation accuracies
        chi_training_acc.append(tree.calc_accuracy(X_train))
        chi_validation_acc.append(tree.calc_accuracy(X_test))

        # Store the depth of the tree
        depth.append(tree.height)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of nodes in a given tree

    Input:
    - node: a node in the decision tree.

    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    if not isinstance(node, DecisionTree):
        raise TypeError("node must be a DecisionTree")
    if not node:
        return 0
    #only node in subtree
    if node.terminal or not node.children:
        return 1

    return 1 + sum(count_nodes(child) for child in node.children)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes