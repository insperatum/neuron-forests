import multiprocessing as mp
from collections import deque, namedtuple
from time import time

from memory_profiler import profile
import numpy as np
import random

from util import *


TreeParameters = namedtuple(
    'TreeParameters',
    ['max_depth', 'stop_when', 'n_node_features', 'n_node_thresholds',
     'training_par_thresholds', 'training_par_features'])
Split = namedtuple(
    'Split',
    ['feature_key', 'threshold', 'entropy', 'cond', 'targets_left', 'targets_right', 'proportion_left', 'proportion_right'])


class Tree:
    def __init__(self, params):
        self.params = params
        self.root = None

    def train(self, features, idxs, targets):
        self.root = _TreeNode(self.params, features, idxs, targets)
        queue = deque([self.root])

        # print "Initial Proportion {}".format(map(lambda p: int(100*p), proportion(targets)))
        current_depth = 0
        while len(queue)>0:
            node = queue.popleft()
            node.make_split()
            split = node.split
            if node.depth>current_depth:
                current_depth = node.depth
                print "Depth {}".format(current_depth)
            # print "Depth {},\tSplit ({}) {}:{} ({})".format(
            #     node.depth,
            #     map(lambda p: int(100*p), proportion(split.targets_left)), len(split.targets_left),
            #     len(split.targets_right), map(lambda p: int(100*p), proportion(split.targets_right)))

            if node.depth < self.params.max_depth:
                if ~self.params.stop_when(split.targets_left):
                    node.left = _TreeNode(self.params, features, node.idxs[:, split.cond], split.targets_left, node.depth + 1)
                    queue.append(node.left)
                if ~self.params.stop_when(split.targets_right):
                    node.right = _TreeNode(self.params, features, node.idxs[:, ~split.cond], split.targets_right, node.depth + 1)
                    queue.append(node.right)

    def predict(self, features, idxs):
        queue = deque([(self.root, np.arange(len(idxs[0])))])
        pred = np.empty((len(idxs[0]), 3), 'd')

        while len(queue)>0:
            node, node_idxs_idxs = queue.popleft()
            feature_vals = features[node.split.feature_key](idxs[:, node_idxs_idxs])
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
    def __init__(self, params, features, idxs, targets, depth=1):
        self.params = params
        self.features = features
        self.idxs = idxs
        self.targets = targets
        self.depth = depth
        self.split = None
        self.left, self.right = None, None

    def make_split(self):
        split_features = [self.features[random.choice(self.features.keys())] for _ in range(self.params.n_node_features)]
        self.split = par_max_by(split_features, self.params.training_par_features, train_feature, (self.params, self.idxs, self.targets), score_split)

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