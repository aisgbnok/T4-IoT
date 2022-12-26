"""
Author: Anthony Swierkosz
This is Anthony's rewrite of the original code.
"""
# All primary imports
import json
import os

import numpy as np
import tqdm
from sklearn.preprocessing import LabelEncoder


# Supress sklearn warnings
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

from sklearn.naive_bayes import MultinomialNB

# Seed Value
# (ensures consistent dataset splitting between runs)
SEED = 1

# Default path for IoT data after unzipping the dataset (in Colab)
ROOT = 'C:/Users/Anthony/Documents/AA4/iot_data'

# Percentage of samples to use for testing (feel free to change)
SPLIT = 0.3

# Port count threshold (e.g., discard port feature values that appear less than N times)
PORT_COUNT = 10

# Maximum and minimum number of samples to allow when loading each IoT device
MAX_SAMPLES_PER_CLASS = 1000
MIN_SAMPLES_PER_CLASS = 20


def load_data(root, min_samples, max_samples):
    """
    Load json feature files produced from feature extraction.

    The device label (MAC) is identified from the directory in which the feature file was found.
    Returns x and y as separate multidimensional arrays.
    The instances in x contain only the first 6 features.
    The ports, domain, and cipher features are stored in separate arrays for easier process in stage 0.

    Parameters
    ----------
    root : str
           Path to the directory containing samples.
    min_samples : int
                  The number of samples each class must have at minimum (else it is pruned).
    max_samples : int
                  Stop loading samples for a class when this number is reached.

    Returns
    -------
    features_misc : numpy array
                    Traffic statistical features.
    features_ports : numpy array
                     Vectorized word-bags (e.g., counts) for ports.
    features_domains : numpy array
                       Vectorized word-bags (e.g., counts) for domains.
    features_ciphers : numpy array
                       Vectorized word-bags (e.g., counts) for ciphers.
    labels : numpy array
             (numerical) Labels for all samples in the dataset.
    """
    x = []
    x_p = []
    x_d = []
    x_c = []
    y = []

    port_dict = dict()
    domain_set = set()
    cipher_set = set()

    # Create paths and do instance count filtering
    f_paths = []
    f_counts = dict()
    for rt, dirs, files in os.walk(root):
        for f_name in files:
            path = os.path.join(rt, f_name)
            label = os.path.basename(os.path.dirname(path))
            name = os.path.basename(path)
            if name.startswith("features") and name.endswith(".json"):
                f_paths.append((path, label, name))
                f_counts[label] = 1 + f_counts.get(label, 0)

    # Load Samples
    processed_counts = {label: 0 for label in f_counts.keys()}
    for fpath in tqdm.tqdm(f_paths):  # Enumerate all sample files
        path = fpath[0]
        label = fpath[1]
        if f_counts[label] < min_samples:
            continue
        if processed_counts[label] >= max_samples:
            continue  # Limit
        processed_counts[label] += 1
        with open(path, "r") as fp:
            features = json.load(fp)
            instance = [features["flow_volume"],
                        features["flow_duration"],
                        features["flow_rate"],
                        features["sleep_time"],
                        features["dns_interval"],
                        features["ntp_interval"]]
            x.append(instance)
            x_p.append(list(features["ports"]))
            x_d.append(list(features["domains"]))
            x_c.append(list(features["ciphers"]))
            y.append(label)
            domain_set.update(list(features["domains"]))
            cipher_set.update(list(features["ciphers"]))
            for port in set(features["ports"]):
                port_dict[port] = 1 + port_dict.get(port, 0)

    # Prune rarely seen ports
    port_set = set()
    for port in port_dict.keys():
        if port_dict[port] > PORT_COUNT:  # Filter out ports that are rarely seen to reduce feature dimensionality
            port_set.add(port)

    # Map to word-bag
    print("Generating word-bags ... ")
    for i in tqdm.tqdm(range(len(y))):
        x_p[i] = list(map(lambda x: x_p[i].count(x), port_set))
        x_d[i] = list(map(lambda x: x_d[i].count(x), domain_set))
        x_c[i] = list(map(lambda x: x_c[i].count(x), cipher_set))

    return np.array(x).astype(float), np.array(x_p), np.array(x_d), np.array(x_c), np.array(y)


def classify_bayes(x_tr, y_tr, x_ts, y_ts):
    """
    Use a multinomial naive bayes classifier to analyze the 'bag of words' seen in the ports/domain/ciphers features.
    Returns the prediction results for the training and testing datasets as an array of tuples in which each row
    represents a data instance and each tuple is composed as the predicted class and the confidence of prediction.

    Parameters
    ----------
    x_tr : numpy array
           Array containing training samples.
    y_tr : numpy array
           Array containing training labels.
    x_ts : numpy array
           Array containing testing samples.
    y_ts : numpy array
           Array containing testing labels.

    Returns
    -------
    c_tr : numpy array
           Prediction results for training samples.
    c_ts : numpy array
           Prediction results for testing samples.
    """
    classifier = MultinomialNB()
    classifier.fit(x_tr, y_tr)

    # Produce class and confidence for training samples
    c_tr = classifier.predict_proba(x_tr)
    c_tr = [(np.argmax(instance), max(instance)) for instance in c_tr]

    # Produce class and confidence for testing samples
    c_ts = classifier.predict_proba(x_ts)
    c_ts = [(np.argmax(instance), max(instance)) for instance in c_ts]

    return c_tr, c_ts


def do_stage_0(xp_tr, xp_ts, xd_tr, xd_ts, xc_tr, xc_ts, y_tr, y_ts):
    """
    Perform stage 0 of the classification procedure:
        process each multinomial feature using naive bayes
        return the class prediction and confidence score for each instance feature

    Parameters
    ----------
    xp_tr : numpy array
           Array containing training (port) samples.
    xp_ts : numpy array
           Array containing testing (port) samples.
    xd_tr : numpy array
           Array containing training (port) samples.
    xd_ts : numpy array
           Array containing testing (port) samples.
    xc_tr : numpy array
           Array containing training (port) samples.
    xc_ts : numpy array
           Array containing testing (port) samples.
    y_tr : numpy array
           Array containing training labels.
    y_ts : numpy array
           Array containing testing labels.

    Returns
    -------
    res_p_tr : numpy array
               Prediction results for training (port) samples.
    res_p_ts : numpy array
               Prediction results for testing (port) samples.
    res_d_tr : numpy array
               Prediction results for training (domains) samples.
    res_d_ts : numpy array
               Prediction results for testing (domains) samples.
    res_c_tr : numpy array
               Prediction results for training (cipher suites) samples.
    res_c_ts : numpy array
               Prediction results for testing (cipher suites) samples.
    """
    # Perform multinomial classification on bag of ports
    res_p_tr, res_p_ts = classify_bayes(xp_tr, y_tr, xp_ts, y_ts)

    # Perform multinomial classification on domain names
    res_d_tr, res_d_ts = classify_bayes(xd_tr, y_tr, xd_ts, y_ts)

    # Perform multinomial classification on cipher suites
    res_c_tr, res_c_ts = classify_bayes(xc_tr, y_tr, xc_ts, y_ts)

    return res_p_tr, res_p_ts, res_d_tr, res_d_ts, res_c_tr, res_c_ts


class Node:
    """
    A node in the decision tree.
    """

    def __init__(self, leaf, c, split, left, right):
        """
        Initialize a node.

        Parameters
        ----------
        leaf : bool
               True if the node is a leaf, False otherwise.
        c : int
            The class of the node.
        split : int
                The feature index to split on.
        left : Node
               The left child node.
        right : Node
                The right child node.
        """
        self.leaf = leaf
        self.c = c
        self.split = split
        self.left = left
        self.right = right


def _split(x, y, feature, value):
    """
    Split the data on a specific split.

    Parameters
    ----------
    x : numpy array
        Array containing samples.
    y : numpy array
        Array containing labels.
    feature : int
              The feature to split on.
    value : int
            The value to split on.

    Returns
    -------
    x_l : numpy array
          Array containing samples of the left subtree.
    x_r : numpy array
          Array containing samples of the right subtree.
    y_l : numpy array
          Array containing labels of the left subtree.
    y_r : numpy array
          Array containing labels of the right subtree.
    """
    x_l = x[x[:, feature] <= value]
    x_r = x[x[:, feature] > value]

    y_l = y[x[:, feature] <= value]
    y_r = y[x[:, feature] > value]

    return x_l, x_r, y_l, y_r


class DecisionTree:
    """
    A decision tree classifier.
    Builds out a decision tree using the gini impurity value as the splitting criterion.
    """

    def __init__(self, max_depth=10, min_node=5):
        self.max_depth = max_depth
        self.min_node = min_node
        self.tree = None
        self.classes = None

    def fit(self, x, y):
        """
        Fit the decision tree classifier to the data.

        Parameters
        ----------
        x : numpy array
            Array containing samples.
        y : numpy array
            Array containing labels.
        """
        self.tree = self._build_tree(x, y, 0)
        self.classes = np.unique(y)

    def _build_tree(self, x, y, depth):
        """
        Build the decision tree recursively.

        Parameters
        ----------
        x : numpy array
            Array containing samples.
        y : numpy array
            Array containing labels.
        depth : int
                Current depth of the tree.

        Returns
        -------
        node : dict
               The node of the decision tree.
        """
        # The branch has reached the max_depth.
        if depth >= self.max_depth:
            return Node(True, self._get_majority_class(y), None, None, None)

        # The number of samples in the group to split is less than the min_node.
        if len(y) <= self.min_node:
            return Node(True, self._get_majority_class(y), None, None, None)

        # All samples belong to the same class
        if len(set(y)) == 1:
            return Node(True, y[0], None, None, None)

        # Find the best split
        split = self._best_split(x, y)

        # The optimal split results in a group with no samples
        if split is None:
            return Node(True, self._get_majority_class(y), None, None, None)

        # Split the data
        x_l, x_r, y_l, y_r = _split(x, y, split)

        # Build the left and right subtrees
        left = self._build_tree(x_l, y_l, depth + 1)
        right = self._build_tree(x_r, y_r, depth + 1)

        # Return the node
        return Node(False, None, split, left, right)

    def _best_split(self, x, y):
        """
        Find best split using gini impurity.

        Parameters
        ----------
        x : numpy array
            Array containing samples.
        y : numpy array
            Array containing labels.

        Returns
        -------
        split : int
                The feature index to split on.
        """
        best_split = None
        best_gini = 1.0

        # Iterate over all features
        for i in range(x.shape[1]):
            # Get the unique values of the feature
            values = np.unique(x[:, i])

            # Iterate over all values
            for v in values:
                # Split the data
                x_l, x_r, y_l, y_r = _split(x, y, i, v)

                # Calculate the gini impurity
                gini = self._gini_impurity(y_l, y_r)

                # Update the best split
                if gini < best_gini:
                    best_split = (i, v)
                    best_gini = gini

        # Return the best split
        return best_split

    def _gini_impurity(self, y_l, y_r):
        """
        Calculate the gini impurity.

        Parameters
        ----------
        y_l : numpy array
              Array containing labels of the left group.
        y_r : numpy array
              Array containing labels of the right group.

        Returns
        -------
        gini : float
               The gini impurity value.
        """
        # Get the number of samples in each group
        n_l = len(y_l)
        n_r = len(y_r)

        # Calculate the gini impurity
        gini = 0.0
        for c in self.classes:
            p_l = np.sum(y_l == c) / n_l
            p_r = np.sum(y_r == c) / n_r
            gini += p_l * (1 - p_l) + p_r * (1 - p_r)

        # Return the gini impurity
        return gini

    def _gini(self, y_l, y_r):
        """
        Calculate the gini impurity of the split.

        Parameters
        ----------
        y_l : numpy array
              Array containing labels of the left subtree.
        y_r : numpy array
              Array containing labels of the right subtree.

        Returns
        -------
        gini : float
                  The gini impurity of the split.
        """
        gini = 0
        for y in [y_l, y_r]:
            # Calculate the probability of each class
            p = [np.sum(y == c) / len(y) for c in self.classes]

            # Calculate the total gini impurity
            gini += (1 - sum([p[i] ** 2 for i in range(len(p))]))

        return gini

    def _get_majority_class(self, y):
        """
        Get the majority class.

        Parameters
        ----------
        y : numpy array
            Array containing labels.

        Returns
        -------
        class : int
                The majority class.
        """
        # Count the number of samples for each class
        counts = np.bincount(y)

        # Return the majority class
        return np.argmax(counts)

    def _predict(self, x, tree):
        """
        Predict the label of a sample.

        Parameters
        ----------
        x : numpy array
            Array containing samples.
        tree : dict
               The tree to use for prediction.

        Returns
        -------
        y : numpy array
            Array containing labels.
        """
        # Make predictions
        y = np.array([self._predict_sample(sample, tree) for sample in x])

        return y

    def _predict_sample(self, x, tree):
        """
        Predict the label of a sample.

        Parameters
        ----------
        x : numpy array
            Array containing samples.
        tree : dict
               The tree to use for prediction.

        Returns
        -------
        y : int
            The label of the sample.
        """
        # Check if the node is a leaf
        if tree["leaf"]:
            return tree["class"]

        # Check if the sample goes left or right
        if x[tree["split"]["feature"]] <= tree["split"]["value"]:
            return self._predict_sample(x, tree["left"])
        else:
            return self._predict_sample(x, tree["right"])

    def _print_tree(self, tree, indent=""):
        """
        Print the tree.

        Parameters
        ----------
        tree : dict
               The tree to print.
        indent : str
                 The current indentation.
        """
        # Check if the node is a leaf
        if tree["leaf"]:
            print(indent + "Predict", tree["class"])
            return

        # Print the split
        print(indent + str(tree["split"]))

        # Print the left and right subtree
        print(indent + "left:")
        self._print_tree(tree["left"], indent + "\t")
        print(indent + "right:")
        self._print_tree(tree["right"], indent + "\t")

    def _plot_tree(self, tree, ax, x_min, x_max, y_min, y_max, depth=0):
        """
        Plot the tree.

        Parameters
        ----------
        tree : dict
               The tree to plot.
        ax : matplotlib axis
             The axis to plot on.
        x_min : float
                The minimum x value.
        x_max : float
                The maximum x value.
        y_min : float
                The minimum y value.
        y_max : float
                The maximum y value.
        depth : int
                The current depth.
        """
        # Check if the node is a leaf
        if tree["leaf"]:
            # Plot the node
            ax.scatter((x_min + x_max) / 2, (y_min + y_max) / 2, c="w", edgecolors="k", s=200, zorder=3)

            # Plot the class

            ax.text((x_min + x_max) / 2, (y_min + y_max) / 2, str(tree["class"]), ha="center", va="center", zorder=4)
            return

        # Get the split
        feature = tree["split"]["feature"]
        value = tree["split"]["value"]

        # Plot the node
        ax.scatter((x_min + x_max) / 2, (y_min + y_max) / 2, c="w", edgecolors="k", s=200, zorder=3)


def gini_impurity(groups, classes):
    """
    Calculate Gini Impurity

    Parameters
    ----------
    groups : tuple
           TODO
    classes : list
           TODO

    Returns
    -------
    res_p_tr : float
               Impurity TODO
    """
    n = sum(len(group) for group in groups)
    impurity = 0.0

    for group in groups:
        if len(group) == 0:
            pass
        else:
            g_score = 0.0
            for c in classes:
                test = [row[-1] for row in group]

                # row[-1] is the class label
                p = [row[-1] for row in group].count(c) / len(group)
                g_score += p * p
            impurity = impurity + (1 - g_score) * (len(group) / n)

    print(impurity)
    return impurity


def split(value, feature, data):
    yes, no = [], []
    for d in data:
        if d[feature] < value:
            no.append(d)
        else:
            yes.append(d)
    return no, yes


def splitter(data, classes):
    index, value, score, groups = 0, 0, 2.0, None
    i = 0
    checked_feats = []
    while i < len(data[0]) - 1:
        for d in data:
            if d[i] not in checked_feats:
                checked_feats.append(d[i])
                groups = split(d[i], i, data)
                gini = gini_impurity(groups, classes)
                if gini < score:
                    index, value, score, groups = i, d[i], gini, groups
        i += 1
        checked_feats = []
    return index, value, score, groups


class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data
        self.split_on = None
        self.split_threshold = None
        self.purity = 2.0

    def set_children(self, feature, value, left_node, right_node):
        self.left = left_node
        self.right = right_node
        self.split_on = feature
        self.split_threshold = value

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def get_purity(self):
        return self.purity

    def print_tree(self, layer=0):
        if self.data is not None:
            print(f"Layer #{layer}: Feature {self.split_on} < {self.split_threshold}")
            layer = layer + 1
            if self.left is not None:
                print(f"Left side of {self.split_on} < {self.split_threshold}")
                self.left.print_tree(layer)
            if self.right is not None:
                print(f"Right side of {self.split_on} < {self.split_threshold}")
                self.right.print_tree(layer)

    def haschildren(self):
        if self.left is not None and self.right is not None:
            return True
        return False

    def getfeaturesplit(self):
        return [self.split_on, self.split_threshold]

    def getdata(self):
        return self.data


def decision_tree(max_depth, min_node, data, classes, num_features, current_node=None, features_used=None):
    #  The branch has reached the max_depth.
    if max_depth == 0:
        return current_node

    if current_node is None:
        current_node = Node(data)
    if features_used is None:
        features_used = []
    elif len(features_used) >= num_features:
        return current_node
    best_split = splitter(data, classes)
    if best_split[2] > current_node.get_purity():
        return current_node
    elif len(best_split[3][0]) == 0 or len(best_split[3][1]) == 0:
        return current_node
    elif len(best_split[3][0]) < min_node or len(best_split[3][1]) < min_node:
        return current_node
    left_node = decision_tree(max_depth - 1, min_node, best_split[3][0], classes, num_features,
                              current_node=current_node.get_left(), features_used=features_used)
    right_node = decision_tree(max_depth - 1, min_node, best_split[3][1], classes, num_features,
                               current_node=current_node.get_right(), features_used=features_used)
    current_node.set_children(best_split[0], best_split[1], left_node, right_node)
    if best_split[0] not in features_used:
        features_used.append(best_split[0])
    return current_node


class RandomForest:

    def __init__(self, n_trees, data_frac, feature_subcount):
        self.n_trees = n_trees
        self.data_frac = data_frac
        self.feature_subcount = feature_subcount
        self.trees = []
        self.num_features = 0

    def fit(self, x, y):
        self.classes = np.unique(y)
        self.num_features = x.shape[1]

        for i in range(self.n_trees):
            x_sample, y_sample = self._get_sample(x, y, True)
            tree = decision_tree(10, 1, x_sample, self.classes, self.feature_subcount)
            self.trees.append(tree)

    def _get_sample(self, x, y, replace=False):
        sample = np.random.choice(x.shape[0], int(x.shape[0] * self.data_frac), replace=replace)
        return x[sample], y[sample]

    def predict(self, data):
        predictions = []
        for d in data:
            predictions.append(self.predict_single(d))
        return predictions

    def predict_single(self, data):
        predictions = []
        for tree in self.trees:
            predictions.append(self.predict_single_tree(data, tree))
        return max(set(predictions), key=predictions.count)

    def predict_single_tree(self, data, tree):
        current_node = tree
        while current_node.haschildren():
            if data[current_node.getfeaturesplit()[0]] < current_node.getfeaturesplit()[1]:
                current_node = current_node.get_left()
            else:
                current_node = current_node.get_right()
        return current_node.getdata()[0][-1]

    def print_trees(self):
        for tree in self.trees:
            tree.print_tree()
            print("")


def random_forest(n_trees, split, feature_sub_count, x_tr, y_tr):
    # Append the class labels to the training data
    x = np.hstack((x_tr, y_tr.reshape(-1, 1)))
    # Get list of unique classes
    classes = np.unique(y_tr)

    # Create a list of trees
    forest = []

    # Create a list of random indices to sample from
    slice_size = int(x.shape[0] * split)
    indices = np.arange(x.shape[0])
    for n in range(n_trees):
        x_sample = []
        slice_index = np.random.choice(indices, size=slice_size, replace=False)

        for si in slice_index:
            x_sample.append(x[si])

        bigtree = decision_tree(5, 10, x_sample, classes, feature_sub_count)
        bigtree.print_tree()
        forest.append(bigtree)
    return forest


def rf_test(x_tr, y_tr, x_ts, y_ts):
    """
    Performs testing on the random forest.

    Parameters
    ----------
    x_tr : numpy array
           Array containing training samples.
    y_tr : numpy array
           Array containing training labels.
    x_ts : numpy array
           Array containing testing samples.
    y_ts : numpy array
           Array containing testing labels.

    Returns
    -------
    accuracy : float
           Accuracy percentage from test run.
    """
    # Train Random Forest
    correct_number = 0
    trees = random_forest(1, .7, 1, x_tr, y_tr)
    print(f'Length: {len(trees)}')

    # Generate predictions for each sample
    for sample in x_ts:
        tree_predictions = {}

        for tree in trees:
            currentnode = tree
            x_class = None
            while currentnode.haschildren():
                split = currentnode.getfeaturesplit()
                sample_val = sample[split[0]]
                if sample_val < split[1]:
                    currentnode = currentnode.get_left()
                else:
                    currentnode = currentnode.get_right()
            class_count = {}
            for d in currentnode.getdata():
                if d[-1] not in class_count.keys():
                    class_count[d[-1]] = 1
                else:
                    class_count[d[-1]] = class_count[d[-1]] + 1

            x_class = max(class_count, key=class_count.get)
            if x_class not in tree_predictions.keys():
                tree_predictions[x_class] = 1
            else:
                tree_predictions[x_class] += 1

        # Check if class prediction was correct
        predicted_class = max(tree_predictions, key=tree_predictions.get)

        x_dex = np.where(x_ts == sample)
        actual_class = y_ts[x_dex[0]][0]

        # print("Predicted class: " + str(predicted_class) + " Actual class: " + str(actual_class))

        if predicted_class == actual_class:
            correct_number += 1
        else:
            print("Predicted class: " + str(predicted_class) + " Actual class: " + str(actual_class))

    return correct_number / len(x_ts)


def do_stage_1(x_tr, x_ts, y_tr, y_ts):
    """
    Perform stage 1 of the classification procedure:
        train a random forest classifier using the NB prediction probabilities

    Parameters
    ----------
    x_tr : numpy array
           Array containing training samples.
    y_tr : numpy array
           Array containing training labels.
    x_ts : numpy array
           Array containing testing samples.
    y_ts : numpy array
           Array containing testing labels.

    Returns
    -------
    pred : numpy array
           Final predictions on testing dataset.
    """

    # Train Random Forest
    rf = RandomForest(10, .7, 1)
    rf.fit(x_tr, y_tr)

    accuracy = rf_test(x_tr, y_tr, x_ts, y_ts)
    return accuracy


def main():
    """
    Load data, encode labels to numeric, and perform classification stages.
    """
    # load dataset
    print("Loading dataset ...")
    x, x_p, x_d, x_c, y = load_data(ROOT, min_samples=MIN_SAMPLES_PER_CLASS,
                                    max_samples=MAX_SAMPLES_PER_CLASS)

    # Encode labels into numerical values
    print("\nEncoding labels ...")
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    print("Dataset Statistics:")
    print(f"\t Classes: {le.classes_.shape[0]}")
    print(f"\t Samples: {y.shape[0]}")
    print("\t Dimensions: ", x.shape, x_p.shape, x_d.shape, x_c.shape)

    # Shuffle
    print(f"\nShuffling dataset using seed {SEED} ...")
    s = np.arange(y.shape[0])
    np.random.seed(SEED)
    np.random.shuffle(s)
    x, x_p, x_d, x_c, y = x[s], x_p[s], x_d[s], x_c[s], y[s]

    # Split
    print(f"Splitting dataset using train:test ratio of {int((1 - SPLIT) * 10)}:{int(SPLIT * 10)} ...")
    cut = int(y.shape[0] * SPLIT)
    x_tr, xp_tr, xd_tr, xc_tr, y_tr = x[cut:], x_p[cut:], x_d[cut:], x_c[cut:], y[cut:]
    x_ts, xp_ts, xd_ts, xc_ts, y_ts = x[:cut], x_p[:cut], x_d[:cut], x_c[:cut], y[:cut]

    # Perform stage 0
    print("\nPerforming Stage 0 classification ...")
    p_tr, p_ts, d_tr, d_ts, c_tr, c_ts = do_stage_0(xp_tr, xp_ts, xd_tr, xd_ts, xc_tr, xc_ts, y_tr, y_ts)

    # Build stage 1 dataset using stage 0 results
    # NB predictions are concatenated to the statistical attributes processed from the flows
    x_tr_full = np.hstack((x_tr, p_tr, d_tr, c_tr))
    x_ts_full = np.hstack((x_ts, p_ts, d_ts, c_ts))

    # Perform final classification
    print("Performing Stage 1 classification ...")
    pred = do_stage_1(x_tr_full, x_ts_full, y_tr, y_ts)

    # Print classification report
    print(f"\nPrediction: {pred}")
    # print(classification_report(y_ts, pred, target_names=le.classes_))


main()
