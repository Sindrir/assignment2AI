import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
from sklearn.datasets import make_blobs

clusters = 4
count = 3000
iterations = 50

def kmean(k, data, count, iter):
    centroid = data
    np.random.shuffle(centroid)
    centroid = centroid[:k,:]
    # centroid = random.sample(data, k)
    # centroid = data[np.random.randint(data.shape[0], size=k)]
    # print(centroid)
    distance = np.zeros(k)
    flag = np.zeros(count)
    shortestCluster = np.zeros(count)
    # print(distance)
    for x in range(iter):
        # meanDistances = np.zeros(shape=data.shape())
        # meanCount = np.zeros(k)
        for i in range(count):
            for n in range(k):
                distance[n] = np.sqrt((centroid[n,0] - data[i,0])**2 + (centroid[n,1] - data[i,1])**2)
                # print("Distance between point " + str(i) + " for cluster " + str(n) + ": " + str(distance[n]))
            shortestCluster[i] = np.argmin(distance)
            # print("Shortest distance for point " + str(i) + " to cluster " + str(shortestCluster[i]))
            for n in range(k):
                if shortestCluster[i] == n:
                    # meanCount[n] += 1
                    flag[i] = n
                centroid[n] = np.mean(data[shortestCluster == n], axis=0)
            # for n in range(k):
            #     centroid[n] = meanDistances[n] / meanCount[n]
        if x == 0:
            colors = cm.hsv(np.linspace(0,1,k+1))
            for i in range(count):
                plt.scatter(data[i,0], data[i,1], color=colors[int(flag[i])], alpha=0.25)
            for i in range(k):
                plt.scatter(centroid[i,0], centroid[i,1], color="black")
            plt.show()
    return flag, centroid

features, target = make_blobs(n_samples=count,
                                n_features=2,
                                centers=clusters,
                                cluster_std=0.6,
                                shuffle=True)
colors = cm.hsv(np.linspace(0,1,clusters+1))
print(colors[1])
flag, centroid = kmean(clusters,features,count, iterations)
for i in range(count):
    plt.scatter(features[i,0], features[i,1], color=colors[int(flag[i])], alpha=0.25)
for i in range(clusters):
    plt.scatter(centroid[i,0], centroid[i,1], color="black")
plt.show()
