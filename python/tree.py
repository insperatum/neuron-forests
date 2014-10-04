import numpy as np
def train(Xfeatures, Y):
	minEntropy = None
	for i in range(0, 100):
		f = np.random.randint(2)
		F = X[f]()
		t = np.random.choice(F)
		a = F < t
		e = entropy(Y[a]) + entropy(Y[~a])
		if(minEntropy is None or e < minEntropy):
			minEntropy = e
			feature = f
			threshold = t
			split = [Y[a], Y[~a]]

	print("entropy: ", minEntropy)
	print("feature: ", feature)
	print("threshold: ", threshold)
	print("split: ", split)


def entropy(list):
	if(list.size == 0):
		return 0
	else:
		p = np.bincount(list) / float(list.size)
		vfunc = np.vectorize(lambda p: 0 if p==0 else -p * np.log(p))
		return np.sum(vfunc(p))