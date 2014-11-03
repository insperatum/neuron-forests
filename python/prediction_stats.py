__author__ = 'luke'

import sys
import os
import numpy as np
from scipy import io

dir = sys.argv[2]
def find_preds(root, path=""):
    dirs = []
    for f in os.listdir(root + "/" + path):
        if os.path.isdir(root + "/" + path + f):
            dirs = dirs + find_preds(root, path + f + "/")
        elif f == "pred.mat":
            dirs = dirs + [root + "/" + path]
    return dirs

dirs = find_preds(dir)
print "{} pred.mat files found (taking 50)".format(len(dirs))
dirs = dirs[1:50]

from comparison_init import *

errs = np.empty(len(dirs))
preds = np.empty((len(dirs), targets_test.shape[0], targets_test.shape[1]))
cumulative = np.zeros(targets_test.shape)

train_time = None
i=1

for d in dirs:
    # print "Loading {}: {}".format(i, f)
    tic = time.time()
    pred = io.loadmat(d + "pred.mat")["pred"]
    # print "took {}s".format(time.time() - tic)
    cumulative[:, 0] += pred[:, :, :, 0].flatten()
    cumulative[:, 1] += pred[:, :, :, 1].flatten()
    cumulative[:, 2] += pred[:, :, :, 2].flatten()
    mean = cumulative/i
    diff = mean - targets_test
    err = np.mean(diff*diff, axis=0)
    errs[i-1] = np.mean(err)
    print "Error with {} tree(s):".format(i), err

    stats = io.loadmat(d + "stats.mat")
    if train_time is None:
        train_time = stats["train_time"] / len(dirs)
        test_time = stats["test_time"] / len(dirs)
        depth_node_counts = stats["depth_node_counts"] / len(dirs)
        depth_example_counts = stats["depth_example_counts"] / len(dirs)
        depth_proportions_avg = stats["depth_proportions_avg"] / len(dirs)
    else:
        train_time += stats["train_time"] / len(dirs)
        test_time += stats["test_time"] / len(dirs)
        depth_node_counts += stats["depth_node_counts"] / len(dirs)
        depth_example_counts += stats["depth_example_counts"] / len(dirs)
        depth_proportions_avg += stats["depth_proportions_avg"] / len(dirs)
    i += 1

print "Errors:", errs
io.savemat(dir + "/summary.mat",
           {"errors":errs, "train_time":train_time, "test_time":test_time,
            "depth_node_counts":depth_node_counts, "depth_example_counts":depth_example_counts,
            "depth_proportions_avg":depth_proportions_avg,
            "pred": mean.reshape(pred.shape)})