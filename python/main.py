import time
# import cPickle as pickle

from Forest import *
from Feature import *
from util import limit_memory
limit_memory(7000)

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
path = "model/{}".format(time.strftime("%d-%m-%y %H:%M:%S"))
sample = False
params = ForestParameters(
    save_path=path,
    n_trees=8,
    bag_proportion=0.5,
    training_par_trees=8,
    testing_par_trees=4,
    tree_params=TreeParameters(
        max_depth=6,
        min_size=100,
        min_proportion=0.01,
        n_node_feature_bases=5,
        n_node_features_total=30,
        n_node_thresholds=10,
        training_par_features=1,
        training_par_thresholds=1))
features = get_feature_generator("features/im1",
        max_offset=1)

forest = Forest(params)
if sample:
    sample = np.random.randint(idxs[0].size, size=100000)
    forest.train(features, idxs[:, sample], targets[sample])
else:
    forest.train(features, idxs, targets)

print "\n\nPREDICTING\n----------"
im = Helmstaedter2013["im"][0, 1]
features = get_feature_generator("features/im2")
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