import multiprocessing as mp
from collections import deque, namedtuple
from time import time

import numpy as np
import random

from Tree import *
from util import *

ForestParameters = namedtuple(
    'ForestParameters',
    ['tree_params', 'n_trees', 'par_trees'])


class Forest:
    def __init__(self, params):
        self.params = params
        self.trees = map(lambda x: Tree(params.tree_params), range(params.n_trees))

    def train(self, features, idxs, targets):
        t = time()
        maybe_par_map(
            train_tree,
            [(i, self.trees[i], features, idxs, targets) for i in range(len(self.trees))], self.params.par_trees)
        print("\nTraining took {} seconds".format(int(time() - t)))

def train_tree(args):
    i, tree, features, idxs, targets = args
    print "\nTraining tree {}\n--------------".format(str(i))
    t = time()
    tree.train(features, idxs, targets)
    print("Tree {} took {} seconds".format(i, int(time() - t)))