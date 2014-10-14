import itertools

import numpy as np
from scipy import ndimage, io

from Forest import *
from util import *
from features import *

print "Loading Helmstaedter2013 data"
Helmstaedter2013 = io.loadmat("data/Helmstaedter_etal_Nature_2013_e2006_TrainingData_all.mat")

print "Initializing"
segTrue = Helmstaedter2013["segTrue"][0, 0].astype(int)
nz = np.nonzero(segTrue)
min_idx = np.min(nz, 1)
max_idx = np.max(nz, 1)
shape = tuple(max_idx - min_idx + 1)
xs, ys, zs = np.ix_(range(min_idx[0], max_idx[0] + 1), range(min_idx[1], max_idx[1] + 1),
                    range(min_idx[2], max_idx[2] + 1))
steps = get_steps(segTrue)
idxs = np.array(np.unravel_index((xs * steps[0] + ys * steps[1] + zs * steps[2]).flatten(), segTrue.shape))
targets = segTrue[tuple(idxs)] != 0

print "\n\nTRAINING\n--------"
sample = True
params = ForestParameters(
    n_trees=2, testing_par_trees=8,
    tree_params=TreeParameters(
        max_depth=2, min_size=100, min_proportion=0.01, n_node_features=15, n_node_thresholds=15,
        training_par_features=8, training_par_thresholds=1))

forest = Forest(params)
features = get_features_dict("features/im1", max_offset=1)
if sample:
    sample = np.random.randint(idxs[0].size, size=100000)
    forest.train(features, idxs[:, sample], targets[sample])
else:
    forest.train(features, idxs, targets)

print "\n\nPREDICTING\n----------"
im = Helmstaedter2013["im"][0, 1]
features = get_features_dict("features/im2", max_offset=1)
shape = im.shape
xs, ys, zs = np.ix_(range(shape[0]), range(shape[1]), range(shape[2]))
idxs = np.array(np.unravel_index((xs * steps[0] + ys * steps[1] + zs * steps[2]).flatten(), segTrue.shape))
pred = forest.predict(features, idxs).reshape(shape)
io.savemat("pred.mat", {'pred': pred})


print "\n\nDISPLAYING\n----------"
from mlabwrap import mlab
seunglab = "/home/luke/Documents/masters/code/seunglab"
mlab.path(mlab.path(), seunglab + "/vis")
mlab.BrowseComponents('ii', im, pred)