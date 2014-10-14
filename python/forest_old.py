import numpy as np
from tree_old import Tree
import pp
import time

job_server = pp.Server()


class Forest:
    def __init__(self, nTrees, minEntropy, maxDepth, nFeatures, nThresholds):
        self.trees = map(lambda x: Tree(minEntropy, maxDepth, nFeatures, nThresholds), range(0, nTrees))

    def train(self, features, idxs, Y):
        outerTime = time.time()
        for i in range(0, len(self.trees)):
            print "Training tree", str(i)
            innerTime = time.time()
            self.trees[i].train(features, idxs, Y)
            print("Tree took {} seconds".format(int(time.time() - innerTime)))
        print("Training took {} seconds".format(int(time.time() - outerTime)))

    def predict(self, features, idxs):
        ncpus = job_server.get_ncpus() - 1
        job_server.set_ncpus(ncpus)
        ntrees = len(self.trees)

        jobs = map(
            lambda tree: job_server.submit(tree.predict, (features, idxs), (), ("numpy as np", "util", "time")),
            self.trees
        )

        outerTime = time.time()
        pred = np.zeros(idxs[0].shape)
        for i in range(0, len(jobs)):
            print "Predicting tree", str(i)
            innerTime = time.time()
            pred += jobs[i]()
            print("Tree took {} seconds".format(int(time.time() - innerTime)))
        pred /= ntrees
        print("Predicting took {} seconds".format(int(time.time() - outerTime)))
        return pred