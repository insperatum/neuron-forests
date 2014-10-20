import multiprocessing as mp
from collections import deque, namedtuple
import time

import numpy as np
import random

from util import *

TreeParameters = namedtuple(
    'TreeParameters',
    ['max_depth', 'min_size', 'min_proportion', 'n_node_feature_bases', 'n_node_features_total', 'n_node_thresholds',
     'training_par_thresholds', 'training_par_features', 'preload_features'])
Split = namedtuple(
    'Split',
    ['feature_key', 'threshold', 'entropy', 'cond', 'targets_left', 'targets_right', 'proportion_left', 'proportion_right'])


# feature_cache = None

class Tree:
    def __init__(self, params):
        self.params = params
        self.root = None

    def train(self, features, idxs, targets):
        self.root = _TreeNode(self.params, idxs, targets)
        if self.params.preload_features:
            features.cache()

        queue = deque([self.root])

        # print "Initial Proportion {}".format(map(lambda p: int(100*p), proportion(targets)))
        current_depth = 0
        feature_cache = None
        start_time = time.time()
        while len(queue)>0:
            node = queue.popleft()
            if node.depth > current_depth:
                if current_depth>0:
                    print "Depth {} took {:.2f}s".format(current_depth, time.time() - start_time)
                    start_time = time.time()
                current_depth = node.depth
                # feature_cache = features.subset(self.params.n_node_feature_bases)
            # print "Depth {},\tSplit ({}) {}:{} ({})".format(
            #     node.depth,
            #     map(lambda p: int(100*p), proportion(split.targets_left)), len(split.targets_left),
            #     len(split.targets_right), map(lambda p: int(100*p), proportion(split.targets_right)))


            node.make_split(features)
            split = node.split

            if node.depth < self.params.max_depth:
                if not stop_when(split.targets_left, self.params.min_size, self.params.min_proportion):
                    node.left = _TreeNode(self.params, node.idxs[:, split.cond], split.targets_left, node.depth + 1)
                    queue.append(node.left)
                if not stop_when(split.targets_right, self.params.min_size, self.params.min_proportion):
                    node.right = _TreeNode(self.params, node.idxs[:, ~split.cond], split.targets_right, node.depth + 1)
                    queue.append(node.right)

        print "Depth {} took {:.2f}s".format(current_depth, time.time() - start_time)

    def predict(self, features, idxs):
        queue = deque([(self.root, np.arange(len(idxs[0])))])
        pred = np.empty((len(idxs[0]), 3), 'd')

        while len(queue)>0:
            node, node_idxs_idxs = queue.popleft()
            feature = features.from_key(node.split.feature_key)
            feature_vals = feature(idxs[:, node_idxs_idxs])
            cond = feature_vals < node.split.threshold
            left_idxs_idxs = node_idxs_idxs[cond]
            right_idxs_idxs = node_idxs_idxs[~cond]

            if node.left is None:
                pred[left_idxs_idxs, :] = node.split.proportion_left
            else:
                queue.append((node.left, left_idxs_idxs))

            if node.right is None:
                pred[right_idxs_idxs, :] = node.split.proportion_right
            else:
                queue.append((node.right, right_idxs_idxs))

        return pred

class _TreeNode:
    def __init__(self, params, idxs, targets, depth=1):
        self.params = params
        self.idxs = idxs
        self.targets = targets
        self.depth = depth
        self.split = None
        self.left, self.right = None, None

    def make_split(self, features):
        gen = features.subset(self.params.n_node_feature_bases)
        split_features = [gen.random() for _ in range(self.params.n_node_features_total)]
        best = par_max_by(split_features, self.params.training_par_features, train_feature, (self.params, self.idxs, self.targets), score_split)
        self.split = best

def stop_when(output, min_size, min_proportion):
    return len(output) < min_size or all(map(lambda p: p<min_proportion or p>1-min_proportion, proportion(output)))

def score_split(split):
    return -split.entropy

def train_feature(feature, params, idxs, targets):
    feature_vals = feature(idxs)
    thresholds = list(np.random.choice(feature_vals, params.n_node_thresholds))
    best = par_max_by(thresholds, params.training_par_thresholds, train_threshold, (targets, feature_vals), score_split)
    return best._replace(feature_key=feature.key)

def train_threshold(threshold, targets, feature_vals):
    cond = feature_vals < threshold
    targets_left = targets[cond]
    targets_right = targets[~cond]
    split_entropy = (len(targets_left) * entropy(targets_left) + len(targets_right) * entropy(targets_right)) / len(targets)
    return Split(
        None, threshold, split_entropy, cond, targets_left, targets_right,
        proportion(targets_left), proportion(targets_right))