from scipy import io
import numpy as np


class Split:
    def __init__(self, entropy, feature, threshold, split, eLeft, eRight, YLeft, YRight):
        self.entropy = entropy
        self.feature = feature
        self.threshold = threshold
        self.split = split
        self.eLeft = eLeft
        self.eRight = eRight
        self.YLeft = YLeft
        self.YRight = YRight


class Feature:
    def __init__(self, root, path, offset=(0, 0, 0)):
        self.root = root
        self.path = path
        self.offset = offset

    def __call__(self, idxs):
        mat = io.loadmat(self.root + "/" + self.path)
        scale = mat["scale"][0, 0]
        im = mat["im"]
        xs = np.maximum(0, np.minimum(im.shape[0] - 1, idxs[0] * scale + self.offset[0])).astype(int)
        ys = np.maximum(0, np.minimum(im.shape[1] - 1, idxs[1] * scale + self.offset[1])).astype(int)
        zs = np.maximum(0, np.minimum(im.shape[2] - 1, idxs[2] * scale + self.offset[2])).astype(int)
        return im[xs, ys, zs]
