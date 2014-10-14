import os
import time

from scipy import io
import numpy as np

from util import *


class Feature:
    def __init__(self, root, path, offset=(0, 0, 0)):
        self.root = root
        self.path = path
        self.offset = offset
        self.key = (self.path, self.offset)

    def __call__(self, idxs_unravelled):
        mat = io.loadmat(self.root + "/" + self.path)
        scale = mat["scale"][0, 0]
        im = mat["im"]
        xs, ys, zs = idxs_unravelled
        xs = np.maximum(0, np.minimum(im.shape[0] - 1, xs * scale + self.offset[0])).astype(int)
        ys = np.maximum(0, np.minimum(im.shape[1] - 1, ys * scale + self.offset[1])).astype(int)
        zs = np.maximum(0, np.minimum(im.shape[2] - 1, zs * scale + self.offset[2])).astype(int)
        return im[xs, ys, zs].flatten()

def get_feature_paths(root, path=""):
    feature_paths = []
    for f in os.listdir(root + "/" + path):
        if os.path.isdir(root + "/" + path + f):
            feature_paths = feature_paths + get_feature_paths(root, path + f + "/")
        else:
            feature_paths = feature_paths + [path + f]
    return feature_paths


def get_features_dict(root, max_offset):
    out = {}
    for p in get_feature_paths(root):
        for o1 in range(-max_offset, max_offset + 1):
            for o2 in range(-max_offset, max_offset + 1):
                for o3 in range(-max_offset, max_offset + 1):
                    offsets = (o1, o2, o3)
                    f = Feature(root, p, offsets)
                    out.update({f.key: f})
    return out