import os
import random
import time

from scipy import io
import numpy as np

from util import *


class FeatureGenerator:
    def __init__(self, root, paths, offsets):
        self.root = root
        self.paths = paths
        self.size = len(self.paths)
        self.offsets = offsets
        self.cached = None


    def from_key(self, key):
        path, offset = key
        return self.gen_feature(path, offset)

    def gen_feature(self, path, offset):
        if self.cached is not None:
            return Feature(self.cached[path], offset)
        else:
            return Feature(FeatureBase(self.root, path), offset)

    def all(self):
        return [self.gen_feature(path, o)
                    for path in self.paths
                    for o in self.offsets]

    def random(self):
        path = random.choice(self.paths)
        o = random.choice(self.offsets)
        return self.gen_feature(path, o)

    def subset(self, n_features):
        gen = FeatureGenerator(self.root, np.random.permutation(self.paths)[:n_features], self.offsets)
        gen.cached = self.cached
        return gen

    def cache(self):
        if self.cached is None:
            start_time = time.time()
            print "Preloading {} features".format(len(self.paths))
            self.cached = {}
            for p in self.paths:
                self.cached.update({p:CachedFeatureBase(FeatureBase(self.root, p))})
            print "Preloading complete (took {} seconds)".format(time.time() - start_time)
        return self

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
        FeatureBase.__init__(self, base.root, base.path)
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

def get_feature_generator(root, offsets = [(0, 0, 0)]):
    return FeatureGenerator(root, _get_feature_paths(root), offsets)