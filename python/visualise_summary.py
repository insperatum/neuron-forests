import sys
from scipy import io
from comparison_init import *
summary_file = sys.argv[2]


summary = io.loadmat(summary_file)
pred = summary["pred"]

print summary["depth_node_counts"].shape

if summary["depth_node_counts"].shape[0] < summary["depth_node_counts"].shape[1]:
    summary["depth_node_counts"] = summary["depth_node_counts"].T
if summary["depth_example_counts"].shape[0] < summary["depth_example_counts"].shape[1]:
    summary["depth_example_counts"] = summary["depth_example_counts"].T
if summary["depth_proportions_avg"].shape[0] < summary["depth_proportions_avg"].shape[1]:
    summary["depth_proportions_avg"] = summary["depth_proportions_avg"].T


print "\n\n\n\n\n-----------------Summary-----------------\n"
print "Training time per tree: {}s".format(summary["train_time"][0,0])
print "Test time per tree: {}s".format(summary["test_time"][0,0])

print "{}|{}|{}|{}".format("Depth".ljust(5), "Avg n nodes".ljust(15), "Avg. n examples".ljust(16), "Avg. proportion")
for i in range(len(summary["depth_node_counts"])):
    print "{}|{}|{}|{}".format(
        str(i).ljust(5),
        "{:3.3f}".format(summary["depth_node_counts"][i, 0]).ljust(15),
        "{:3.3f}".format(summary["depth_example_counts"][i, 0]).ljust(16),
        "{:3.3f}".format(summary["depth_proportions_avg"][i, 0])
    )



diff0 = (pred[:, :, :, 0].flatten() - targets_test[:, 0])
diff1 = (pred[:, :, :, 1].flatten() - targets_test[:, 1])
diff2 = (pred[:, :, :, 2].flatten() - targets_test[:, 2])
err1 = np.mean(diff0 * diff0)
err2 = np.mean(diff1 * diff1)
err3 = np.mean(diff2 * diff2)
error = (err1 + err2 + err3)/3

print "\nMean square error: {}".format(error)

print "\n\nDISPLAYING\n----------"



from mlabwrap import mlab
seunglab = "/home/luke/Documents/masters/code/seunglab"
matlabpath = "/home/luke/neuron-forests/matlab"
mlab.path(mlab.path(), seunglab + "/vis")
mlab.path(mlab.path(), seunglab + "/segmentation")
mlab.path(mlab.path(), matlabpath + "/vis")
# mlab.BrowseComponents('ii', targets_test.reshape(pred.shape).astype(float), pred)
mlab.vis2(im[tuple(idxs_test)].reshape((pred.shape[0], pred.shape[1], pred.shape[2])),
         targets_test.reshape(pred.shape).astype(float), pred, 0.85)

# mlab.path(mlab.path(), seunglab + "/analysis")
# mlab.plot_rand_error(pred, targets_test.reshape(pred.shape).astype(float))
