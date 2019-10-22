import numpy as np
from sklearn.datasets import make_blobs
features, target = make_blobs(n_samples=250,
                                n_features=2,
                                centers=6,
                                cluster_std=1.3,
                                shuffle=True)
# f = open("syntheticData.txt", "w")
# f.write(features)
# f.close()
#features.tofile("syntheticData", sep=",")
np.save("syntheticData", features)
