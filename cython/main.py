import numpy as np
import tree
X = np.array([
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1],
	[2, 0],
	[2, 3],
	[1, 3],
	[3, 2],
	[3, 1],
	[2, 0]])
Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0])
tree.make(X, Y)