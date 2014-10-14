import multiprocessing as mp
from collections import deque, namedtuple
from time import time

from memory_profiler import profile
import numpy as np
import random

from util import *


TreeParameters = namedtuple(
    'TreeParameters',
    ['max_depth', 'min_size', 'min_proportion', 'n_node_features', 'n_node_thresholds', 'par_thresholds', 'par_features'])
Split = namedtuple(
    'Split',
    ['feature', 'threshold', 'entropy', 'cond', 'targets_left', 'targets_right'])


class Tree:
    def __init__(self, params):
        self.params = params
        self.root = None

    def train(self, features, idxs, targets):
        self.root = _TreeNode(self.params, features, idxs, targets)
        queue = deque([self.root])

        print "Initial Proportion {:.2f}".format(proportion(targets))

        while len(queue)>0:
            node = queue.popleft()
            node.make_split()
            split = node.split
            print "Depth {},\tSplit ({:.2f}) {}:{} ({:.2f})".format(
                node.depth, proportion(split.targets_left), len(split.targets_left),
                len(split.targets_right), proportion(split.targets_right))

            if node.depth < self.params.max_depth:
                if len(split.targets_left) > self.params.min_size and self.params.min_proportion < proportion(split.targets_left) < 1-self.params.min_proportion:
                    node.left = _TreeNode(self.params, features, node.idxs[:, split.cond], split.targets_left, node.depth + 1)
                    queue.append(node.left)
                if len(split.targets_right) > self.params.min_size and self.params.min_proportion  < proportion(split.targets_right) < 1-self.params.min_proportion:
                    node.right = _TreeNode(self.params, features, node.idxs[:, ~split.cond], split.targets_right, node.depth + 1)
                    queue.append(node.right)

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
        t = time()
        split_features = [self.features[random.choice(self.features.keys())] for _ in range(self.params.n_node_features)]
        self.split = par_max_by(split_features, self.params.par_features, test_feature, (self.params, self.idxs, self.targets), score_split)
        print time()-t

def score_split(split):
    return -split.entropy

def test_feature(feature, params, idxs, targets):
    feature_vals = feature(idxs)
    thresholds = list(np.random.choice(feature_vals, params.n_node_thresholds))
    best = par_max_by(thresholds, params.par_thresholds, test_threshold, (targets, feature_vals), score_split)
    return best._replace(feature=feature)

def test_threshold(threshold, targets, feature_vals):
    cond = feature_vals < threshold
    targets_left = targets[cond]
    targets_right = targets[~cond]
    split_entropy = (len(targets_left) * entropy(targets_left) + len(targets_right) * entropy(targets_right)) / len(targets)
    return Split(None, threshold, split_entropy, cond, targets_left, targets_right)