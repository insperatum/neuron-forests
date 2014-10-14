import multiprocessing as mp
from collections import deque, namedtuple
import time
import cPickle as pickle
import os

import numpy as np
import random

from Tree import *
from util import *

ForestParameters = namedtuple(
    'ForestParameters',
    ['tree_params', 'n_trees', 'testing_par_trees', 'training_par_trees', 'save_path'])


class Forest:
    def __init__(self, params):
        self.params = params
        self.tree_keys = None
        if not os.path.exists(self.params.save_path): os.mkdir(self.params.save_path)

    def train(self, features, idxs, targets):
        start_time = time.time()
        self.tree_keys = maybe_par_map(
            train_tree,
            [(i, self.params, features, idxs, targets) for i in range(self.params.n_trees)],
            self.params.training_par_trees)
        print("\nTraining complete. Forest took {} seconds".format(int(time.time() - start_time)))
        pickle.dump(self, open(self.params.save_path + "/Forest.pkl", "wb"), -1)

    def predict(self, features, idxs):
        s = par_sum(
            zip(range(len(self.tree_keys)), self.tree_keys),
            self.params.testing_par_trees,
            predict_tree,
            (features, idxs))
        return s / self.params.n_trees

def train_tree(args):
    i, params, features, idxs, targets = args
    start_time = time.time()
    print "Training tree {}".format(str(i))
    tree = Tree(params.tree_params)
    tree.train(features, idxs, targets)
    print("Tree {} took {} seconds".format(i, int(time.time() - start_time)))
    file = params.save_path + "/Tree " + str(i) + ".pkl"
    pickle.dump(tree, open(file, "wb"), -1)
    return file

def predict_tree(args, features, idxs):
    i, tree_key = args
    tree = pickle.load(open(tree_key))
    print "Predicting tree {}".format(str(i))
    return tree.predict(features, idxs)