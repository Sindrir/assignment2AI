import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def kmean(k, data, count, iter):
    centroid = data
    np.random.shuffle(centroid)
    centroid = centroid[:k,:]
    distance = np.zeros(k)
    flag = np.zeros(count)
    shortestCluster = np.zeros(count)
    for x in range(iter):
        meanX = np.zeros(k)
        meanY = np.zeros(k)
        meanCount = np.zeros(k)
        for i in range(count):
            for n in range(k):
                tmp = 0
                print(len(data[0,:]))
                for feature in range(len(data[0,:])):
                    tmp += (centroid[n,feature] - data[i,feature])**2
                distance[n] = np.sqrt(tmp)
            shortestCluster[i] = np.argmin(distance)
            for n in range(k):
                if shortestCluster[i] == n:
                    flag[i] = n
        if x == 0:
            colors = cm.hsv(np.linspace(0,1,k+1))
            for i in range(count):
                plt.scatter(data[i,0], data[i,1], color=colors[int(flag[i])], alpha=0.25)
            for i in range(k):
                plt.scatter(centroid[i,0], centroid[i,1], color="black")
            plt.show()
        for i in range(count):
            for n in range(k):
                if shortestCluster[i] == n:
                    meanX[n] += data[i, 0]
                    meanY[n] += data[i, 1]
                    meanCount[n] += 1
        for n in range(k):
            centroid[n,0] = meanX[n]/meanCount[n]
            centroid[n,1] = meanY[n]/meanCount[n]
    return flag, centroid
