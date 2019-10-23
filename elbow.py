from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import iris
import synth

irisD = load_iris()

sseIris = {}
sseSynth = {}
synthData = np.load("syntheticData.npy")

for k in range(1,10):
    kmeansIris = KMeans(n_clusters=k).fit(irisD.data[:,:2])
    sseIris[k] = kmeansIris.inertia_
    kmeansSynth = KMeans(n_clusters=k).fit(synthData)
    sseSynth[k] = kmeansSynth.inertia_
plt.figure()
plt.plot(list(sseIris.keys()), list(sseIris.values()))
plt.xlabel("Clusters")
plt.ylabel("SSE")
plt.show()
plt.figure()
plt.plot(list(sseSynth.keys()), list(sseSynth.values()))
plt.xlabel("Clusters")
plt.ylabel("SSE")
plt.show()

iris.run(2, 150, 50, 2)
iris.run(3, 150, 50, 2)
synth.run(3, 250, 50, synthData)
synth.run(5, 250, 50, synthData)
