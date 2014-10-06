import numpy as np
class tree:
	def train(self, features, idxs, Y, depth=0, percent=0):
		#print "Training on ", len(Y), "examples(", str(np.sum(Y)), "true,", str(len(Y) - np.sum(Y)), "false )"
		print "{}%".format(percent)	
		self.splitEntropy = None
		for i in range(0, 5):
			f = np.random.choice(range(0, len(features)))
			#print "feature: ", f.path
			F = features[f](idxs)
			for j in range(0, 10):
				t = np.random.choice(F)
				a = F < t
				l = Y[a]
				r = Y[~a]
				eLeft = entropy(l)
				eRight = entropy(r)
				e = (len(l)*eLeft + len(r)*eRight)/len(Y)
				if(self.splitEntropy is None or e < self.splitEntropy):
					self.splitEntropy = e
					self.feature = f
					self.threshold = t
					self.split = [a, ~a]
					self.eLeft = eLeft
					self.eRight = eRight
					self.YLeft = l
					self.YRight = r
		#print "\nSplit with feature: ", features[self.feature].path
		#print "         threshold: ", self.threshold
		#print "        split entropy: ", self.splitEntropy
		#print " individual entropies: ", entropy(Y[self.split[0]]), entropy(Y[self.split[1]])

		self.left = None
		if(self.eLeft > 0.2 and depth<3):
			self.left = tree()
			self.left.train(features, idxs[:, self.split[0]], self.YLeft, depth+1, percent)
		self.right = None
		if(self.eRight > 0.2 and depth<3):
			self.right = tree()
			self.right.train(features, idxs[:, self.split[1]], self.YRight, depth+1, percent + pow(2, -1-depth))

	def predict(self, features, idxs, depth=0, percent=0):
		print "{}%".format(percent)	
		a = np.array(features[self.feature](idxs) < self.threshold)
		out = np.empty(idxs[0].shape)
		if any(a):
			if self.left is None: l = np.sum(self.YLeft) / float(len(self.YLeft))
			else: 
				l = self.left.predict(features, idxs[:, a], depth+1, percent)
			out[a] = l
		if any(~a):
			if self.right is None: r = np.sum(self.YRight) / float(len(self.YRight))
			else: 
				r = self.right.predict(features, idxs[:, ~a], depth+1, percent + pow(2, -1-depth))
			out[~a] = r
		return out



def entropy(list):
	if(list.size == 0):
		return 0
	else:
		p = np.bincount(list) / float(list.size)
		vfunc = np.vectorize(lambda p: 0 if p==0 else -p * np.log(p))
		return np.sum(vfunc(p))