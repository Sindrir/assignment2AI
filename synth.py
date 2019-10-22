import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import kmean

def run(clusters, count, iterations, features):
    colors = cm.hsv(np.linspace(0,1,clusters+1))
    flag, centroid = kmean.kmean(clusters,features,count, iterations)
    for i in range(count):
        plt.scatter(features[i,0], features[i,1], color=colors[int(flag[i])], alpha=0.25)
    for i in range(clusters):
        plt.scatter(centroid[i,0], centroid[i,1], color="black")
    plt.show()
    return features, centroid, flag
