from scipy import io
from Forest import *
import sys
import cPickle as pickle

data_file = sys.argv[1]
features_dir = sys.argv[2]
#cmdparams = {'train_frac':0.2, 'offsets':[-2, 0, 2], 'depth':14, 'thresholds':1, 'max_features':'sqrt'}
cmdparams = eval(sys.argv[3])

print "lukeforest"
print cmdparams

from comparison_init import *

offsets = cmdparams["offsets"]
multioffsets = [(o1, o2, o3) for o1 in offsets for o2 in offsets for o3 in offsets]
features = get_feature_generator(features_dir + "/im1", offsets=multioffsets)

print "Training idxs:", min_idx, "to", max_idx_train
print "Test idxs:", min_idx_test, "to", max_idx
print "\n-----------------\n"

if not os.path.exists("model"): os.mkdir("model")
path = "model/{}".format(time.strftime("%d-%m-%y %H:%M:%S"))

params = ForestParameters(
    save_path=path,
    n_trees=1,
    bag_proportion=0.5,
    training_par_trees=1,
    testing_par_trees=1,
    preload_features=True,
    tree_params=TreeParameters(
        max_depth=cmdparams["depth"],
        min_size=2,
        min_proportion=0.01,
        n_node_feature_bases=30,
        n_node_features_total=-1 if cmdparams["max_features"]==None else int(np.sqrt(30 * len(multioffsets))),
        n_node_thresholds=cmdparams["thresholds"],
        training_par_features=1,
        training_par_thresholds=1,
        preload_features=True))

print "\nTraining on {} examples".format(idxs_train[0].size)
tic = time.time()
forest = Forest(params)
forest.train(features, idxs_train, targets_train)
train_time = time.time()-tic
print "Training took {}s".format(train_time)


print "\n\nPredicting on remaining {} examples".format(idxs_test[0].size)
tic = time.time()
shape = max_idx - min_idx_test + 1
pred = forest.predict(features, idxs_test).reshape((shape[0], shape[1], shape[2], 3))
test_time = time.time()-tic
print "Testing took {}s".format(test_time)


diff0 = (pred[:, :, :, 0].flatten() - targets_test[:, 0])
diff1 = (pred[:, :, :, 1].flatten() - targets_test[:, 1])
diff2 = (pred[:, :, :, 2].flatten() - targets_test[:, 2])
print "Mean square errors:"
print np.mean(diff0 * diff0)
print np.mean(diff1 * diff1)
print np.mean(diff2 * diff2)
print "Complete."


t = pickle.load(open(forest.tree_keys[0]))
depth_node_counts = np.zeros(cmdparams['depth'] + 1)
depth_example_counts = np.zeros(cmdparams['depth'] + 1)
depth_proportions_sum = np.zeros(cmdparams['depth'] + 1)

def traverse(node):
    d = node.depth
    depth_node_counts[d] += 1
    depth_example_counts[d] += node.idxs.shape[1]
    depth_proportions_sum[d] += np.mean(0.5 - np.abs(np.array(node.proportion)-0.5))
    if node.left is not None: traverse(node.left)
    if node.right is not None: traverse(node.right)
traverse(t.root)

depth_proportions_avg = depth_proportions_sum/depth_node_counts

print "Depth node counts:\n", depth_node_counts
print "Depth example counts:\n", depth_example_counts
print "Depth avg proportions:\n", depth_proportions_avg

io.savemat("pred.mat", {"pred": pred})
io.savemat("stats.mat", {"train_time": train_time, "test_time":test_time, "depth_node_counts":depth_node_counts,
                         "depth_example_counts":depth_example_counts, "depth_proportions_avg":depth_proportions_avg})


