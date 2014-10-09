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
		job_server.set_ncpus(ncpus)
		ntrees = len(self.trees)

		jobs = map(
			lambda tree: job_server.submit(tree.predict, (features, idxs), (), ("numpy as np", "util")),
			self.trees
			)
		
		pred = np.zeros(idxs[0].shape)
		for i in range(0, len(jobs)):
			pred += jobs[i]()
			print "Predicted tree", str(i)

		preds = map(lambda job: job(), jobs)
		pred = np.sum(preds, 0)/ntrees
		return pred