import os
import random

from scipy import io
import numpy as np

from util import *


class FeatureGenerator:
    def __init__(self, root, paths, max_offset):
        self.root = root
        self.paths = paths
        self.max_offset = max_offset
        self.cached = None

    def from_key(self, key):
        path, offset = key
        return Feature(FeatureBase(self.root, path), offset)

    def random(self):
        path = random.choice(self.paths)
        o1 = random.randint(-self.max_offset, self.max_offset)
        o2 = random.randint(-self.max_offset, self.max_offset)
        o3 = random.randint(-self.max_offset, self.max_offset)
        return Feature(FeatureBase(self.root, path), (o1, o2, o3))

    def subset(self, n_features):
        return FeatureGenerator(self.root, [random.choice(self.paths) for _ in range(n_features)], self.max_offset)

    def cache(self):
        self.cached = map(lambda p: CachedFeatureBase(FeatureBase(self.root, p)), self.paths)

    def clear_cache(self):
        self.cached = None


class FeatureBase:
    def __init__(self, root, path):
        self.root = root
        self.path = path

    def load_data(self):
        return io.loadmat(self.root + "/" + self.path)

class CachedFeatureBase(FeatureBase):
    def __init__(self, base):
        FeatureBase.__init__(self, base.root, base.path, base.offset)
        self.mat = io.loadmat(self.root + "/" + self.path)

    def load_data(self):
        return self.mat

class Feature:
    def __init__(self, base, offset):
        self.base = base
        self.offset = offset
        self.key = (base.path, offset)

    def __call__(self, idxs_unravelled):
        mat = self.base.load_data()
        scale = mat["scale"][0, 0]
        im = mat["im"]
        xs, ys, zs = idxs_unravelled
        xs = np.maximum(0, np.minimum(im.shape[0] - 1, xs * scale + self.offset[0])).astype(int)
        ys = np.maximum(0, np.minimum(im.shape[1] - 1, ys * scale + self.offset[1])).astype(int)
        zs = np.maximum(0, np.minimum(im.shape[2] - 1, zs * scale + self.offset[2])).astype(int)
        return im[xs, ys, zs].flatten()



def _get_feature_paths(root, path=""):
    feature_paths = []
    for f in os.listdir(root + "/" + path):
        if os.path.isdir(root + "/" + path + f):
            feature_paths = feature_paths + _get_feature_paths(root, path + f + "/")
        else:
            feature_paths = feature_paths + [path + f]
    return feature_paths

def get_feature_generator(root, max_offset=None):
    return FeatureGenerator(root, _get_feature_paths(root), max_offset)