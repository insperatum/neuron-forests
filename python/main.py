import numpy as np
import os
from forest import Forest
import itertools
from scipy import ndimage, io
import util


def get_feature_paths(root, path=""):
    feature_paths = []
    for f in os.listdir(root + "/" + path):
        if os.path.isdir(root + "/" + path + f):
            feature_paths = feature_paths + get_feature_paths(root, path + f + "/")
        else:
            feature_paths = feature_paths + [path + f]
    return feature_paths


def get_features_dict(root, max_offset=3):
    out = {}
    for p in get_feature_paths(root):
        for o1 in range(-max_offset, max_offset + 1):
            for o2 in range(-max_offset, max_offset + 1):
                for o3 in range(-max_offset, max_offset + 1):
                    max_offset = (o1, o2, o3)
                    out.update({(p, max_offset): util.Feature(root, p, max_offset)})
    return out


print "Loading Helmstaedter2013 data"
Helmstaedter2013 = io.loadmat("data/Helmstaedter_etal_Nature_2013_e2006_TrainingData_all.mat")

print "Initializing"
segTrue = Helmstaedter2013["segTrue"][0, 0].astype(int)
nz = np.nonzero(segTrue)
minIdx = np.min(nz, 1)
maxIdx = np.max(nz, 1)
shape = tuple(maxIdx - minIdx + 1)


def offsetFunc(idx): return [idx[0] + minIdx[0], idx[1] + minIdx[1], idx[2] + minIdx[2]]  # is tuple(np.add(_)) faster?


idxs = np.array(list(itertools.imap(offsetFunc, np.ndindex(shape)))).T
Y = segTrue[minIdx[0]:maxIdx[0] + 1, minIdx[1]:maxIdx[1] + 1, minIdx[2]:maxIdx[2] + 1] != 0

# print "Training"
# dt = tree()
# dt.train(features, idxs[:,1:100000], Y.flatten()[1:100000])

# print "Predicting"
# im = Helmstaedter2013["im"][0, 0]
# features = np.vectorize(feature)(get_feature_paths("features/im1"))
# shape = im.shape
# idxs = np.array(list(np.ndindex(shape))).T
# pred = dt.predict(features, idxs).reshape(shape)
# io.savemat("pred.mat", {'pred':pred})
# print "Complete."

maxOffset = 1

print "Training"
df = Forest(nTrees=1, minEntropy=0.05, maxDepth=1, nFeatures=30, nThresholds=10)
features = get_features_dict("features/im1", max_offset=maxOffset)

# Sampling
sample = np.random.randint(idxs.shape[1], size=100000)
df.train(features, idxs[:, sample], Y.flatten()[sample])

# No sampling
# df.train(features, idxs, Y.flatten())

print "Predicting"
im = Helmstaedter2013["im"][0, 1]
features = get_features_dict("features/im2", max_offset=maxOffset)
shape = im.shape
idxs = np.array(list(np.ndindex(shape))).T
pred = df.predict(features, idxs).reshape(shape)
io.savemat("pred.mat", {'pred': pred})
print "Complete."

from mlabwrap import mlab

seunglab = "/home/luke/Documents/masters/code/seunglab"
mlab.path(mlab.path(), seunglab + "/vis")
mlab.BrowseComponents('ii', im, pred)