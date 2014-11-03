import numpy as np
from scipy import ndimage, misc, io
import os
import sys

data_file = sys.argv[1]

# def getFiltered(img, scale):
# conv = ndimage.filters.convolve
# def convFull(A, B):
# padding = [(d-1, 0) for d in B.shape]
# 	return conv(np.pad(A, padding, mode='constant'), B, mode='constant')
# dx = [[0.5, -0.5]]
# dy = [[0.5], [-0.5]]
# if(img.ndim == 2): ord1 = [np.array(dx), np.array(dy)]
# else: ord1 = [ np.array([dx]), np.array([dy]), np.array([[[0.5]],[[-0.5]]]) ]
# ord2 = [convFull(ord1[i], ord1[j]) for i in range(0, len(ord1)) for j in range(i, len(ord1))]
# filters = ord1 + ord2
#
# scaled = ndimage.zoom(img, scale)
#return [scaled] + [conv(scaled, f) for f in filters]


def makeFeatures(img, name):
    print("Creating features for " + name)
    if not os.path.exists("features"): os.mkdir("features")
    if not os.path.exists("features/{}".format(name)): os.mkdir("features/{}".format(name))

    if img.ndim == 2:
        orders = [[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]]
    else:
        orders = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 2], [0, 1, 1], [1, 0, 1], [0, 2, 0], [1, 1, 0],
                  [2, 0, 0]]

    for scale in [1., 0.5, 0.25]:
        print("...at scale " + str(scale))
        folder = ("features/{}/scale_{}".format(name, scale))
        if not os.path.exists(folder): os.mkdir(folder)

        scaled = ndimage.zoom(img, scale)

        for o in orders:
            suffix = "_" + "".join(str(x) for x in o)
            feature = ndimage.filters.gaussian_filter(img, 1, o)
            # if(img.ndim == 2):
            # 	filename = "{}/{}{}.png".format(folder, name, suffix)
            # 	misc.imsave(filename, feature)
            filename = "{}/DoG{}.mat".format(folder, suffix)
            io.savemat(filename, {"im": feature, "scale": scale})


print "Loading Helmstaedter2013 data"
Helmstaedter2013 = io.loadmat(data_file)
print "keys", Helmstaedter2013.keys()
a = Helmstaedter2013["im"]
print a.shape
b = a[0, 1]
print b.shape
for i in range(0, 2):
    makeFeatures(Helmstaedter2013["im"][0, i], "im" + str(i + 1))



    # filtered = filter(cat, 0.5)
    # l = int(np.sqrt(len(filtered)))+1
    # for i in range(0, len(filtered)):
    # 	pyplot.subplot(l, l, i+1)
    # 	pyplot.imshow(filtered[i], cmap = pyplot.cm.gray)
    # 	misc.imsave("filtered/%s.png"%i, filtered[i])

    # pyplot.show()