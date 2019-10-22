import matplotlib.cm as cm
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import kmean

clusters = 3
count = 150
iterations = 100

iris = sklearn.datasets.load_iris()
data = iris.data[:,:4]
colors = cm.hsv(np.linspace(0,1,clusters+1))
flag, centroid = kmean.kmean(clusters,data,count, iterations)
for i in range(count):
    plt.scatter(data[i,0], data[i,1], color=colors[int(flag[i])], alpha=0.25)
for i in range(clusters):
    plt.scatter(centroid[i,0], centroid[i,1], color="black")
plt.show()
