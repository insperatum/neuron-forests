from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from scipy import io
from Feature import *
from util import limit_memory
import time
import sys

data_file = sys.argv[1]
features_dir = sys.argv[1]

print "sklearn Random Forest"

print "Loading Helmstaedter2013 data"
Helmstaedter2013 = io.loadmat(data_file)

print "Initializing"
segTrue = Helmstaedter2013["segTrue"][0, 0].astype(int)
nz = np.nonzero(segTrue)
min_idx = np.min(nz, 1)
max_idx = np.max(nz, 1) - 1

max_idx_train = np.array([int((min_idx[0]+max_idx[0])/2), max_idx[1], max_idx[2]])
min_idx_test = np.array([int((min_idx[0]+max_idx[0])/2+1), min_idx[1], min_idx[2]])

idxs_train = get_image_idxs(segTrue, min_idx=min_idx, max_idx=max_idx_train)
targets_train = get_target_affinities(segTrue, idxs_train)

idxs_test = get_image_idxs(segTrue, min_idx=min_idx_test, max_idx=max_idx)
targets_test = get_target_affinities(segTrue, idxs_test)

features = get_feature_generator(features_dir + "/im1", max_offset=0)

print "Training idxs:", min_idx, "to", max_idx_train
print "Test idxs:", min_idx_test, "to", max_idx
print "\n-----------------\n"

print "Training"
#perm = np.random.permutation(idxs[0].size)
# idxs_train = idxs[:, perm[:n]]
# idxs_test = idxs[:, perm[n+1:]]
# targets_train = targets[perm[:n]]
# targets_test = targets[perm[n+1:]]

X = np.empty((idxs_train[0].size, len(features.paths)), dtype=float)
for i in range(0, len(features.paths)):
    X[:, i] = features.gen_feature(features.paths[i], (0, 0, 0))(idxs_train)



print "\nTraining on {} examples".format(idxs_train[0].size)
tic = time.time()
clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
clf.fit(X, targets_train)
print "Training took {}s".format(time.time()-tic)


print "\n\nPredicting on remaining {} examples".format(idxs_test[0].size)
X = np.empty((idxs_test[0].size, len(features.paths)), dtype=float)
for i in range(0, len(features.paths)):
    X[:, i] = features.gen_feature(features.paths[i], (0, 0, 0))(idxs_test)
tic = time.time()
pred = clf.predict_proba(X)
print "Testing took {}s".format(time.time()-tic)


diff0 = (pred[0][:, 1] - targets_test[:, 0])
diff1 = (pred[1][:, 1] - targets_test[:, 1])
diff2 = (pred[2][:, 1] - targets_test[:, 2])
print "Mean square errors:"
print np.mean(diff0 * diff0)
print np.mean(diff1 * diff1)
print np.mean(diff2 * diff2)
print "Complete."



io.savemat("pred.mat", {"pred": pred})