import numpy as np
import random
import util
from memory_profiler import profile
import pp

job_server = pp.Server()


class Tree:
    def __init__(self, minEntropy, maxDepth, nFeatures, nThresholds, depth=0, nComplete=0, nTotal=-1):
        self.minEntropy = minEntropy
        self.maxDepth = maxDepth
        self.nFeatures = nFeatures
        self.nThresholds = nThresholds
        self.depth = depth
        self.nComplete = nComplete
        self.nTotal = nTotal
        self.split = None
        self.left = None
        self.right = None

    def train(self, features, idxs, Y):
        if self.nTotal == -1: self.nTotal = len(Y)
        if self.depth < 3: print "{}%".format(float(100 * self.nComplete) / self.nTotal)
        # print "Training on ", len(Y), "examples(", str(np.sum(Y)), "true,", str(len(Y) - np.sum(Y)), "false )"
        def tryFeature(f, F, Y, nThresholds):
            best = None
            for j in range(0, nThresholds):
                t = random.choice(F)
                a = F < t
                l = Y[a]
                r = Y[~a]
                eLeft = entropy(l)
                eRight = entropy(r)
                e = (len(l) * eLeft + len(r) * eRight) / len(Y)
                if (best is None or e < best.entropy):
                    best = util.Split(
                        entropy=e,
                        feature=f,
                        threshold=t,
                        split=[a, ~a],
                        eLeft=eLeft,
                        eRight=eRight,
                        YLeft=l,
                        YRight=r)
            return best

        def makeJob():
            f = random.choice(features.keys())
            F = features[f](idxs)
            return job_server.submit(tryFeature, (f, F, Y, self.nThresholds), (entropy,),
                                     ("numpy as np", "util", "random"))

        jobs = map(lambda i: makeJob(), range(0, self.nFeatures))
        splits = map(lambda job: job(), jobs)
        self.split = min(splits, key=lambda s: s.entropy)

        if self.split.eLeft > self.minEntropy and self.depth < self.maxDepth:
            self.left = Tree(self.minEntropy, self.maxDepth, self.nFeatures, self.nThresholds, self.depth + 1,
                             self.nComplete, self.nTotal)
            self.left.train(features, idxs[:, self.split.split[0]], self.split.YLeft)
        if self.split.eRight > self.minEntropy and self.depth < self.maxDepth:
            self.right = Tree(self.minEntropy, self.maxDepth, self.nFeatures, self.nThresholds, self.depth + 1,
                              self.nComplete + len(self.split.YLeft), self.nTotal)
            self.right.train(features, idxs[:, self.split.split[1]], self.split.YRight)

    @profile
    def predict(self, features, idxs):
        if self.depth < 3: print "{}%".format(float(100 * self.nComplete) / self.nTotal)
        a = np.array(features[self.split.feature](idxs) < self.split.threshold)
        out = np.zeros(idxs[0].shape)

        if any(a):
            if self.left is None:
                l = np.sum(self.split.YLeft) / float(len(self.split.YLeft))
            else:
                l = self.left.predict(features, idxs[:, a])
            out[a] = l
        if any(~a):
            if self.right is None:
                r = np.sum(self.split.YRight) / float(len(self.split.YRight))
            else:
                r = self.right.predict(features, idxs[:, ~a])
            out[~a] = r

        return out


def entropy(lst):
    if lst.size == 0:
        return 0
    else:
        p = np.bincount(lst) / float(lst.size)
        vfunc = np.vectorize(lambda x: 0 if x == 0 else -x * np.log(x))
        return np.sum(vfunc(p))