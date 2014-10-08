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