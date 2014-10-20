import multiprocessing as mp
from collections import deque, namedtuple
import time
import cPickle as pickle
import os
import gc

from memory_profiler import profile
import numpy as np
import random

from Tree import *
from util import *

ForestParameters = namedtuple(
    'ForestParameters',
    ['tree_params', 'n_trees', 'testing_par_trees', 'training_par_trees', 'save_path', 'bag_proportion', 'preload_features'])


class Forest:
    def __init__(self, params):
        self.params = params
        self.tree_keys = None
        if not os.path.exists(self.params.save_path): os.mkdir(self.params.save_path)

    @profile
    def train(self, features, idxs, targets):
        if self.params.preload_features:
            features.cache()
        start_time = time.time()
        bagged_idxs_idxs = [np.random.permutation(idxs.shape[1])[:idxs.shape[1]*self.params.bag_proportion]
                            for _ in range(self.params.n_trees)]

        self.tree_keys = maybe_par_map(
            train_tree,
            [(i, self.params, features,
              idxs[:, bagged_idxs_idxs[i]],
              targets[bagged_idxs_idxs[i]])
             for i in range(self.params.n_trees)],
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

# @profile
def train_tree(args):
    i, params, features, idxs, targets = args
    start_time = time.time()
    print "Training tree {}".format(str(i))
    tree = Tree(params.tree_params)
    tree.train(features, idxs, targets)
    print("Tree {} took {} seconds".format(i, int(time.time() - start_time)))
    file = params.save_path + "/Tree " + str(i) + ".pkl"
    pickle.dump(tree, open(file, "wb"), -1)
    # tree = None
    # gc.collect()
    return file

def predict_tree(args, features, idxs):
    i, tree_key = args
    tree = pickle.load(open(tree_key))
    print "Predicting tree {}".format(str(i))
    return tree.predict(features, idxs)