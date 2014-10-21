from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from Feature import *
from util import limit_memory
limit_memory(6000)

n = 400000
print "n={}".format(n)

print "Loading Helmstaedter2013 data"
Helmstaedter2013 = io.loadmat("data/Helmstaedter_etal_Nature_2013_e2006_TrainingData_all.mat")

print "Initializing"
segTrue = Helmstaedter2013["segTrue"][0, 0].astype(int)
nz = np.nonzero(segTrue)
min_idx = np.min(nz, 1)
max_idx = np.max(nz, 1) - 1

idxs = get_image_idxs(segTrue, min_idx=min_idx, max_idx=max_idx)
targets = get_target_affinities(segTrue, idxs)
features = get_feature_generator("features/im1", max_offset=0)

perm = np.random.permutation(idxs[0].size)
idxs_train = idxs[:, perm[:n]]
idxs_test = idxs[:, perm[n+1:]]
targets_train = targets[perm[:n]]
targets_test = targets[perm[n+1:]]


X = np.empty((idxs_train[0].size, len(features.paths)), dtype=float)
for i in range(0, len(features.paths)):
    X[:, i] = features.gen_feature(features.paths[i], (0, 0, 0))(idxs_train)

print "\nTraining"
clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
clf.fit(X, targets_train)

print "\nPredicting on same examples"
pred = clf.predict_proba(X)
diff0 = (pred[0][:, 1] - targets_train[:, 0])
diff1 = (pred[1][:, 1] - targets_train[:, 1])
diff2 = (pred[2][:, 1] - targets_train[:, 2])
print "Mean square errors:"
print np.mean(diff0 * diff0)
print np.mean(diff1 * diff1)
print np.mean(diff2 * diff2)


print "\n\nPredicting on remaining {} examples".format(idxs_test[0].size)
X = np.empty((idxs_test[0].size, len(features.paths)), dtype=float)
for i in range(0, len(features.paths)):
    X[:, i] = features.gen_feature(features.paths[i], (0, 0, 0))(idxs_test)

pred = clf.predict_proba(X)
diff0 = (pred[0][:, 1] - targets_test[:, 0])
diff1 = (pred[1][:, 1] - targets_test[:, 1])
diff2 = (pred[2][:, 1] - targets_test[:, 2])
print "Mean square errors:"
print np.mean(diff0 * diff0)
print np.mean(diff1 * diff1)
print np.mean(diff2 * diff2)


print "Complete."