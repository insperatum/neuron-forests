from scipy import io

class Split:
	def __init__(self, entropy, feature, threshold, split, eLeft, eRight, YLeft, YRight):
		self.entropy = entropy
		self.feature = feature
		self.threshold = threshold
		self.split = split
		self.eLeft = eLeft
		self.eRight = eRight
		self.YLeft = YLeft
		self.YRight = YRight

class Feature:
	def __init__(self, root, path):
		self.root = root
		self.path = path
	def __call__(self, idxs):
		mat = io.loadmat(self.root + "/" + self.path)
		scale = mat["scale"][0,0]
		im = mat["im"]
		return im[(idxs[0]*scale).astype(int), (idxs[1]*scale).astype(int), (idxs[2]*scale).astype(int)]
