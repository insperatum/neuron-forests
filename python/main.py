import numpy as np
import os
import tree
import itertools
from scipy import ndimage, io

def get_feature_paths(at):
	feature_paths = []
	for f in os.listdir(at):
		if(os.path.isdir(at + "/" + f)):
			feature_paths = feature_paths + get_feature_paths(at + "/" + f)
		else:
			feature_paths = feature_paths + [at + "/" + f]
	return feature_paths

class feature:
	def __init__(self, path): self.path = path
	def __call__(self, idxs):
		mat = io.loadmat(self.path)
		scale = mat["scale"][0,0]
		im = mat["im"]
		return im[(idxs[0]*scale).astype(int), (idxs[1]*scale).astype(int), (idxs[2]*scale).astype(int)]

feature_paths = get_feature_paths("features/im1")
features = np.vectorize(feature)(feature_paths)
print "feature paths: ", feature_paths
print ""
print "Loading Helmstaedter2013 data"
Helmstaedter2013 = io.loadmat("data/Helmstaedter_etal_Nature_2013_e2006_TrainingData_all.mat")

print "Initializing"
segTrue = Helmstaedter2013["segTrue"][0, 0].astype(int)
nz = np.nonzero(segTrue)
minIdx = np.min(nz, 1)
maxIdx = np.max(nz, 1)
shape = tuple(maxIdx - minIdx + 1)
def offsetFunc(idx): return [idx[0]+minIdx[0], idx[1]+minIdx[1], idx[2]+minIdx[2]] #is tuple(np.add(_)) faster?
idxs = np.array(list(itertools.imap(offsetFunc, np.ndindex(shape)))).T
Y = segTrue[minIdx[0]:maxIdx[0]+1, minIdx[1]:maxIdx[1]+1, minIdx[2]:maxIdx[2]+1]!=0

print "Training"
tree = reload(tree)
dt = tree.tree()
dt.train(features, idxs[:,1:100000], Y.flatten()[1:100000])

print "Predicting"
im = Helmstaedter2013["im"][0, 0]
features = np.vectorize(feature)(get_feature_paths("features/im1"))
shape = im.shape
idxs = np.array(list(np.ndindex(shape))).T
pred = tree.predict(dt, features, idxs).reshape(shape)
io.savemat("pred.mat", {'pred':pred})
print "Complete."