import sys
from scipy import io
from comparison_init import *
summary_file1 = sys.argv[2]
summary_file2 = sys.argv[4]

summary1 = io.loadmat(summary_file1)
pred1 = summary1["pred"]

print summary1["depth_node_counts"].shape

if summary1["depth_node_counts"].shape[0] < summary1["depth_node_counts"].shape[1]:
    summary1["depth_node_counts"] = summary1["depth_node_counts"].T
if summary1["depth_example_counts"].shape[0] < summary1["depth_example_counts"].shape[1]:
    summary1["depth_example_counts"] = summary1["depth_example_counts"].T
if summary1["depth_proportions_avg"].shape[0] < summary1["depth_proportions_avg"].shape[1]:
    summary1["depth_proportions_avg"] = summary1["depth_proportions_avg"].T


print "\n\n\n\n\n-----------------Summary 1-----------------\n"
print "Training time per tree: {}s".format(summary1["train_time"][0,0])
print "Test time per tree: {}s".format(summary1["test_time"][0,0])

print "{}|{}|{}|{}".format("Depth".ljust(5), "Avg n nodes".ljust(15), "Avg. n examples".ljust(16), "Avg. proportion")
for i in range(len(summary1["depth_node_counts"])):
    print "{}|{}|{}|{}".format(
        str(i).ljust(5),
        "{:3.3f}".format(summary1["depth_node_counts"][i, 0]).ljust(15),
        "{:3.3f}".format(summary1["depth_example_counts"][i, 0]).ljust(16),
        "{:3.3f}".format(summary1["depth_proportions_avg"][i, 0])
    )

diff0 = (pred1[:, :, :, 0].flatten() - targets_test[:, 0])
diff1 = (pred1[:, :, :, 1].flatten() - targets_test[:, 1])
diff2 = (pred1[:, :, :, 2].flatten() - targets_test[:, 2])
err1 = np.mean(diff0 * diff0)
err2 = np.mean(diff1 * diff1)
err3 = np.mean(diff2 * diff2)
error = (err1 + err2 + err3)/3

print "\nMean square error: {}".format(error)






#---------------------------------------------



summary2 = io.loadmat(summary_file2)
pred2 = summary2["pred"]

print summary2["depth_node_counts"].shape

if summary2["depth_node_counts"].shape[0] < summary2["depth_node_counts"].shape[1]:
    summary2["depth_node_counts"] = summary2["depth_node_counts"].T
if summary2["depth_example_counts"].shape[0] < summary2["depth_example_counts"].shape[1]:
    summary2["depth_example_counts"] = summary2["depth_example_counts"].T
if summary2["depth_proportions_avg"].shape[0] < summary2["depth_proportions_avg"].shape[1]:
    summary2["depth_proportions_avg"] = summary2["depth_proportions_avg"].T


print "\n-----------------Summary 2-----------------\n"
print "Training time per tree: {}s".format(summary2["train_time"][0,0])
print "Test time per tree: {}s".format(summary2["test_time"][0,0])

print "{}|{}|{}|{}".format("Depth".ljust(5), "Avg n nodes".ljust(15), "Avg. n examples".ljust(16), "Avg. proportion")
for i in range(len(summary2["depth_node_counts"])):
    print "{}|{}|{}|{}".format(
        str(i).ljust(5),
        "{:3.3f}".format(summary2["depth_node_counts"][i, 0]).ljust(15),
        "{:3.3f}".format(summary2["depth_example_counts"][i, 0]).ljust(16),
        "{:3.3f}".format(summary2["depth_proportions_avg"][i, 0])
    )

diff0 = (pred2[:, :, :, 0].flatten() - targets_test[:, 0])
diff1 = (pred2[:, :, :, 1].flatten() - targets_test[:, 1])
diff2 = (pred2[:, :, :, 2].flatten() - targets_test[:, 2])
err1 = np.mean(diff0 * diff0)
err2 = np.mean(diff1 * diff1)
err3 = np.mean(diff2 * diff2)
error = (err1 + err2 + err3)/3

print "\nMean square error: {}".format(error)

from mlabwrap import mlab
seunglab = "/home/luke/Documents/masters/code/seunglab"
matlabpath = "/home/luke/neuron-forests/matlab"
mlab.path(mlab.path(), seunglab + "/vis")
mlab.path(mlab.path(), seunglab + "/segmentation")
mlab.path(mlab.path(), matlabpath + "/vis")
mlab.BrowseComponents('iiii', targets_test.reshape(pred1.shape).astype(float), pred1, pred2)

# print pred1.shape
# print pred2.shape
# print targets_test.reshape(pred1.shape).astype(float)

#mlab.vis_compare(targets_test.reshape(pred1.shape).astype(float), pred1, pred2, 0.8)

# mlab.path(mlab.path(), seunglab + "/analysis")
# mlab.plot_rand_error(pred, targets_test.reshape(pred.shape).astype(float))
