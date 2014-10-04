import numpy as np
import os
import tree
from scipy import ndimage, io

feature_paths = []
root = "features/im1"
def recurse_folders(at):
	global feature_paths
	for f in os.listdir(at):
		if(os.path.isdir(at + "/" + f)):
			recurse_folders(at + "/" + f)
		else:
			feature_paths = feature_paths + [at + "/" + f]
recurse_folders(root)

def getFeature(path):
	def get() :
		mat = io.loadmat(path)
		scale = mat["scale"][0,0]
		return ndimage.zoom(mat["im"], 1/scale)[1:225, 1:225, 1:225]
	return get
features = np.vectorize(getFeature)(feature_paths)
print(features[0]())
# tree.make(features, Y)