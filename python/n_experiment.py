import sys

from Forest import *
from Feature import *
from util import limit_memory
limit_memory(6000)

n = int(sys.argv[1])
#n = 1000
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

perm = np.random.permutation(idxs[0].size)
idxs_train = idxs[:, perm[:n]]
idxs_test = idxs[:, perm[n+1:]]
targets_train = targets[perm[:n]]
targets_test = targets[perm[n+1:]]

print "Creating Forest"
path = "model/n_experiment_{}".format(n)
params = ForestParameters(
    save_path=path,
    n_trees=20,
    bag_proportion=0.5,
    training_par_trees=1,
    testing_par_trees=1,
    preload_features=True,
    tree_params=TreeParameters(
        max_depth=3,
        min_size=100,
        min_proportion=0.01,
        n_node_feature_bases=5,
        n_node_features_total=30,
        n_node_thresholds=10,
        training_par_features=1,
        training_par_thresholds=1,
        preload_features=True))
features = get_feature_generator("features/im1",
        max_offset=1)

forest = Forest(params)
print "\nTraining on {} data examples".format(idxs_train[0].size)
forest.train(features, idxs_train, targets_train)

print "\nPredicting on same examples"
pred = forest.predict(features, idxs_train)
diff = pred - targets_train
err = np.mean(diff * diff)
print "Mean square error = {}".format(err)



print "\n\nPredicting on remaining {} examples".format(idxs_test[0].size)
pred = forest.predict(features, idxs_test)
diff = pred - targets_test
err = np.mean(diff * diff)
print "Mean square error = {}".format(err)

print "\n\nSaving results".format(targets_test.size)
io.savemat(path + "/results.mat", {'targets':targets_train, 'pred': pred})

print "Complete."