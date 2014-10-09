import numpy as np
from tree import tree
import pp
job_server = pp.Server()

class forest:
	def __init__(self, nTrees, minEntropy, maxDepth, nFeatures, nThresholds):
		self.trees = map(lambda x: tree(minEntropy, maxDepth, nFeatures, nThresholds), range(0,nTrees))

	def train(self, features, idxs, Y):
		for i in range(0, len(self.trees)):
			print "Training tree", str(i)
			self.trees[i].train(features, idxs, Y)

	def predict(self, features, idxs):
		ncpus = job_server.get_ncpus() - 1
		ntrees = len(self.trees)

		jobTrees = map(
			lambda i: map(
				lambda j: self.trees[j],
				range(ntrees * i / ncpus, ntrees * (i+1) / ncpus)),
			range(0, ncpus))

		jobs = map(lambda trees: job_server.submit(self.sumPreds, (trees, idxs, features), (), ("numpy as np", "util")), jobTrees)
		preds = map(lambda job: job(), jobs)
		pred = np.sum(preds, 0)/ntrees
		print pred.shape
		return pred

	def sumPreds(self, trees, idxs, features):
		pred = np.zeros(idxs[0].shape)
		for i in range(0, len(trees)):
			print "Predicting tree", str(i)
			pred += trees[i].predict(features, idxs)
		return pred