import numpy as np
def make(X, Y):

	minEntropy = None
	for i in range(0, 100):
		f = np.random.randint(X.shape[1])
		F = X[:, f]
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
		def elemEntropy(p): return 0 if p==0 else -p * np.log(p)
		vfunc = np.vectorize(elemEntropy)
		return np.sum(vfunc(p))