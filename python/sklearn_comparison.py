from scipy import io
from Forest import *
import sys
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import cPickle as pickle


data_file = sys.argv[1]
features_dir = sys.argv[2]
#cmdparams = {'train_frac':0.2, 'depth':14, 'type':'RandomForestClassifier', 'max_features':'sqrt'}
cmdparams = eval(sys.argv[3])
print cmdparams

if cmdparams["type"]=="RandomForestClassifier":
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=cmdparams["depth"], max_features=cmdparams["max_features"])
elif cmdparams["type"]=="ExtraTreesClassifier":
    clf = ExtraTreesClassifier(n_estimators=50, n_jobs=-1, max_depth=cmdparams["depth"], max_features=cmdparams["max_features"])

print "sklearn_forest"

from comparison_init import *

features = get_feature_generator(features_dir + "/im1")
fs = features.all()

print "Training idxs:", min_idx, "to", max_idx_train
print "Test idxs:", min_idx_test, "to", max_idx
print "\n-----------------\n"

# if not os.path.exists("model"): os.mkdir("model")
X = np.empty((idxs_train[0].size, len(features.paths)), dtype=float)
for i in range(len(fs)):
    X[:, i] = fs[i](idxs_train)

print "\nTraining on {} examples".format(idxs_train[0].size)
tic = time.time()
clf.fit(X, targets_train)
train_time = time.time()-tic
print "Training took {}s".format(train_time)

# file = "model/Forest.pkl"
# pickle.dump(clf, open(file, 'wb'), -1)

print "\n\nPredicting on remaining {} examples".format(idxs_test[0].size)
X = np.empty((idxs_test[0].size, len(features.paths)), dtype=float)
for i in range(len(fs)):
    X[:, i] = fs[i](idxs_test)

tic = time.time()
shape = max_idx - min_idx_test + 1
pred = np.array(clf.predict_proba(X))[:, :, 1].T.reshape((shape[0], shape[1], shape[2], 3))
test_time = time.time() - tic
print "Testing took {}s".format(test_time)








print "Calculating stats!"
def get_stats(tree):
    depth_node_counts = np.zeros(tree.max_depth + 1)
    depth_example_counts = np.zeros(tree.max_depth + 1)
    depth_proportions_sum = np.zeros(tree.max_depth + 1)
    def traverse(i, depth):
        depth_node_counts[depth] += 1
        depth_example_counts[depth] += tree.n_node_samples[i]
        if tree.children_left[i] == -1:
            val = tree.value[i]
        else:
            value_left = traverse(tree.children_left[i], depth+1)
            value_right = traverse(tree.children_right[i], depth+1)
            val = value_left + value_right
        prop = val[:, 1].astype(float) / np.sum(val, axis=1)
        depth_proportions_sum[depth] += np.mean(0.5 - np.abs(prop-0.5))
        return val
    traverse(0, 0)
    depth_proportions_avg = depth_proportions_sum/depth_node_counts
    return depth_node_counts, depth_example_counts, depth_proportions_avg

trees = clf.estimators_
depth_node_counts = None
for e in trees:
    dnc, dec, dpa = get_stats(e.tree_)
    if depth_node_counts is None:
        depth_node_counts = dnc / len(trees)
        depth_example_counts = dec / len(trees)
        depth_proportion_avg = dpa / len(trees)
    else:
        depth_node_counts += dnc / len(trees)
        depth_example_counts += dec / len(trees)
        depth_proportion_avg += dpa / len(trees)

print "Depth node counts:\n", depth_node_counts
print "Depth example counts:\n", depth_example_counts
print "Depth avg proportions:\n", depth_proportion_avg





diff0 = (pred[:, :, :, 0].flatten() - targets_test[:, 0])
diff1 = (pred[:, :, :, 1].flatten() - targets_test[:, 1])
diff2 = (pred[:, :, :, 2].flatten() - targets_test[:, 2])
print "Mean square errors:"
print np.mean(diff0 * diff0)
print np.mean(diff1 * diff1)
print np.mean(diff2 * diff2)
print "Complete."


io.savemat("summary.mat", {"pred":pred, "train_time": train_time/50, "test_time":test_time/50, "depth_node_counts":depth_node_counts,
                         "depth_example_counts":depth_example_counts, "depth_proportions_avg":depth_proportion_avg})