from scipy import io
from Feature import *
import time
import sys

data_file = sys.argv[1]
if len(sys.argv)>3: cmdparams = eval(sys.argv[3])

print "Loading Helmstaedter2013 data"
tic = time.time()
Helmstaedter2013 = io.loadmat(data_file)
print "Loading took {}s".format(time.time() - tic)

print "Initializing"
segTrue = Helmstaedter2013["segTrue"][0, 0].astype(int)
im = Helmstaedter2013["im"][0, 0]

nz = np.nonzero(segTrue)
min_idx = np.min(nz, 1)
max_idx = np.max(nz, 1) - 1

frac = cmdparams["train_frac"]
max_idx_train = np.array([min_idx[0] + int((max_idx[0] - min_idx[0])*frac), max_idx[1], max_idx[2]])
min_idx_test = np.array([min_idx[0] + int((max_idx[0] - min_idx[0])*frac)+1, min_idx[1], min_idx[2]])

idxs_train = get_image_idxs(segTrue, min_idx=min_idx, max_idx=max_idx_train)
targets_train = get_target_affinities(segTrue, idxs_train)

idxs_test = get_image_idxs(segTrue, min_idx=min_idx_test, max_idx=max_idx)
targets_test = get_target_affinities(segTrue, idxs_test)
