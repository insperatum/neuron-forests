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
max_idx = np.max(nz, 1) - 1

idxs = get_image_idxs(segTrue, min_idx=min_idx, max_idx=max_idx)
targets = get_target_affinities(segTrue, idxs)


print "\n\nTRAINING\n--------"
sample = True
def stop_when(output): return len(output) < 100 or all(map(lambda p: p<0.01 or p>0.99, proportion(output)))
params = ForestParameters(
    n_trees=5,
    testing_par_trees=4,
    tree_params=TreeParameters(
        max_depth=6,
        stop_when=stop_when,
        n_node_features=20,
        n_node_thresholds=10,
        training_par_features=mp.cpu_count(),
        training_par_thresholds=1))
features = get_features_dict("features/im1",
        max_offset=1)

forest = Forest(params)
if sample:
    sample = np.random.randint(idxs[0].size, size=100000)
    forest.train(features, idxs[:, sample], targets[sample])
else:
    forest.train(features, idxs, targets)


print "\n\nPREDICTING\n----------"
im = Helmstaedter2013["im"][0, 1]
features = get_features_dict("features/im2", max_offset=1)
shape = (im.shape[0]-1, im.shape[1]-1, im.shape[2]-1)
max_idx = (shape[0]-1, shape[1]-1, shape[2]-1)
idxs = get_image_idxs(segTrue, max_idx)

pred = forest.predict(features, idxs).reshape((shape[0], shape[1], shape[2], 3))
io.savemat("pred.mat", {'pred': pred})


print "\n\nDISPLAYING\n----------"
from mlabwrap import mlab
seunglab = "/home/luke/Documents/masters/code/seunglab"
matlabpath = "/home/luke/Documents/masters/code/neuron-forests/matlab"
mlab.path(mlab.path(), seunglab + "/vis")
mlab.path(mlab.path(), seunglab + "/segmentation")
mlab.path(mlab.path(), matlabpath + "/vis")
mlab.vis(im, pred, 0.9)