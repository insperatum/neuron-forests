import numpy as np
import util
import pp
job_server = pp.Server()

class tree:
	def __init__(self, minEntropy, maxDepth, nFeatures, nThresholds, depth = 0, nComplete = 0, nTotal = -1):
		self.minEntropy = minEntropy
		self.maxDepth = maxDepth
		self.nFeatures = nFeatures
		self.nThresholds = nThresholds
		self.depth = depth
		self.nComplete = nComplete
		self.nTotal = nTotal

	def train(self, features, idxs, Y):
		if(self.nTotal==-1): self.nTotal = len(Y)
		if(self.depth < 3): print "{}%".format( float(100 * self.nComplete) / self.nTotal )
		#print "Training on ", len(Y), "examples(", str(np.sum(Y)), "true,", str(len(Y) - np.sum(Y)), "false )"
		def tryFeature(f, F, Y, nThresholds):
			best = None
			for j in range(0, nThresholds):
				t = np.random.choice(F)
				a = F < t
				l = Y[a]
				r = Y[~a]
				eLeft = entropy(l)
				eRight = entropy(r)
				e = (len(l)*eLeft + len(r)*eRight)/len(Y)
				if(best is None or e < best.entropy):
					best = util.Split(
						entropy = e,
						feature = f,
						threshold = t,
						split = [a, ~a],
						eLeft = eLeft,
						eRight = eRight,
						YLeft = l,
						YRight = r)
			return best

		def makeJob():
			f = np.random.choice(features.keys())
			F = features[f](idxs)
			return job_server.submit(tryFeature, (f, F, Y, self.nThresholds), (entropy,), ("numpy as np", "util"))
		jobs = map(lambda i: makeJob(), range(0, self.nFeatures))
		splits = map(lambda job: job(), jobs)
		self.split = min(splits, key=lambda s: s.entropy)

		self.left = None
		if(self.split.eLeft > self.minEntropy and self.depth<self.maxDepth):
			self.left = tree(self.minEntropy, self.maxDepth, self.nFeatures, self.nThresholds, self.depth+1, self.nComplete, self.nTotal)
			self.left.train(features, idxs[:, self.split.split[0]], self.split.YLeft)
		self.right = None
		if(self.split.eRight > self.minEntropy and self.depth<self.maxDepth):
			self.right = tree(self.minEntropy, self.maxDepth, self.nFeatures, self.nThresholds, self.depth+1, self.nComplete + len(self.split.YLeft), self.nTotal)
			self.right.train(features, idxs[:, self.split.split[1]], self.split.YRight)

	def predict(self, features, idxs):
		if(self.depth < 3): print "{}%".format( float(100 * self.nComplete) / self.nTotal )
		a = np.array(features[self.split.feature](idxs) < self.split.threshold)
		out = np.empty(idxs[0].shape)
		if any(a):
			if self.left is None: l = np.sum(self.split.YLeft) / float(len(self.split.YLeft))
			else: l = self.left.predict(features, idxs[:, a])
		if any(~a):
			if self.right is None: r = np.sum(self.split.YRight) / float(len(self.split.YRight))
			else: r = self.right.predict(features, idxs[:, ~a])
			
		out[a] = l
		out[~a] = r
		return out

def entropy(list):
	if(list.size == 0):
		return 0
	else:
		p = np.bincount(list) / float(list.size)
		vfunc = np.vectorize(lambda p: 0 if p==0 else -p * np.log(p))
		return np.sum(vfunc(p))