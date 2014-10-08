import numpy as np
import Split
import pp
job_server = pp.Server()

class tree:
	def __init__(self, depth = 0, nComplete = 0, nTotal = -1):
		self.depth = depth
		self.nComplete = nComplete
		self.nTotal = nTotal

	def train(self, features, idxs, Y):
		if(self.nTotal==-1): self.nTotal = len(Y)
		if(self.depth < 3): print "{}%".format( float(self.nComplete) / self.nTotal )
		#print "Training on ", len(Y), "examples(", str(np.sum(Y)), "true,", str(len(Y) - np.sum(Y)), "false )"
		def tryFeature(f, F, Y, n):
			best = None
			for j in range(0, n):
				t = np.random.choice(F)
				a = F < t
				l = Y[a]
				r = Y[~a]
				eLeft = entropy(l)
				eRight = entropy(r)
				e = (len(l)*eLeft + len(r)*eRight)/len(Y)
				if(best is None or e < best.entropy):
					best = Split.Split(
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
			return job_server.submit(tryFeature, (f, F, Y, 100), (entropy,), ("numpy as np", "Split"))
		jobs = map(lambda i: makeJob(), range(0, 4))
		splits = map(lambda job: job(), jobs)
		self.split = min(splits, key=lambda s: s.entropy)

		#print "\nSplit with feature: ", features[self.feature].path
		#print "         threshold: ", self.threshold
		#print "        split entropy: ", self.splitEntropy
		#print " individual entropies: ", entropy(Y[self.split[0]]), entropy(Y[self.split[1]])

		self.left = None
		if(self.split.eLeft > 0.05 and self.depth<5):
			self.left = tree(self.depth+1, self.nComplete, self.nTotal)
			self.left.train(features, idxs[:, self.split.split[0]], self.split.YLeft)
		self.right = None
		if(self.split.eRight > 0.05 and self.depth<5):
			self.right = tree(self.depth+1, self.nComplete + len(self.split.YLeft), self.nTotal)
			self.right.train(features, idxs[:, self.split.split[1]], self.split.YRight)

	def predict(self, features, idxs):
		if(self.depth < 3): print "{}%".format( float(self.nComplete) / self.nTotal )
		a = np.array(features[self.feature](idxs) < self.threshold)
		out = np.empty(idxs[0].shape)
		if any(a):
			if self.left is None: l = np.sum(self.YLeft) / float(len(self.YLeft))
			else: 
				l = self.left.predict(features, idxs[:, a])
			out[a] = l
		if any(~a):
			if self.right is None: r = np.sum(self.YRight) / float(len(self.YRight))
			else: 
				r = self.right.predict(features, idxs[:, ~a])
			out[~a] = r
		return out

def entropy(list):
	if(list.size == 0):
		return 0
	else:
		p = np.bincount(list) / float(list.size)
		vfunc = np.vectorize(lambda p: 0 if p==0 else -p * np.log(p))
		return np.sum(vfunc(p))