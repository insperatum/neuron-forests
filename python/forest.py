import numpy as np
from tree import tree
class forest:
	def __init__(self, n):
		self.trees = map(lambda x: tree(), range(0,n))

	def train(self, features, idxs, Y):
		for i in range(0, len(self.trees)):
			print "Training tree", str(i)
			self.trees[i].train(features, idxs, Y)

	def predict(self, features, idxs):
		pred = np.zeros(idxs[0].shape)
		for i in range(0, len(self.trees)):
			print "Predicting tree", str(i)
			pred += self.trees[i].predict(features, idxs) / len(self.trees)
		return pred