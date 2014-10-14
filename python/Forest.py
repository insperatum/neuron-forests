import multiprocessing as mp
from collections import deque, namedtuple
import time

import numpy as np
import random

from Tree import *
from util import *

ForestParameters = namedtuple(
    'ForestParameters',
    ['tree_params', 'n_trees', 'testing_par_trees'])


class Forest:
    def __init__(self, params):
        self.params = params
        self.trees = map(lambda x: Tree(params.tree_params), range(params.n_trees))

    def train(self, features, idxs, targets):
        t = time.time()
        map( #Can't par yet, because trees don't actually get modified
            train_tree,
            [(i, self.trees[i], features, idxs, targets) for i in range(len(self.trees))])
        print("\nTraining took {} seconds".format(int(time.time() - t)))

    def predict(self, features, idxs):
        s = par_sum(
            zip(range(len(self.trees)), self.trees),
            self.params.testing_par_trees,
            predict_tree,
            (features, idxs))
        return s / len(self.trees)

def train_tree(args):
    i, tree, features, idxs, targets = args
    print "\nTraining tree {}\n--------------".format(str(i))
    t = time.time()
    tree.train(features, idxs, targets)
    print("Tree {} took {} seconds".format(i, int(time.time() - t)))

def predict_tree(args, features, idxs):
    i, tree = args
    print "Predicting tree {}".format(str(i))
    return tree.predict(features, idxs)