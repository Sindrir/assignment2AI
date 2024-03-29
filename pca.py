import numpy as np
import matplotlib.pyplot as plt

def imageCompression(image, comp = 100):
    centered = image - np.mean(image, axis=1)
    eVal, eVec = np.linalg.eig(np.cov(centered))
    vecSize = np.size(eVec,axis=1)
    index = np.argsort(eVal)
    index = index[::-1]
    eVec = eVec[:,index]
    eVal = eVal[index]
    components = comp
    if (components > 0 and components < vecSize):
        eVec = eVec[:,range(components)]
    score = np.dot(eVec.T, centered)
    recon = np.dot(eVec, score) + np.mean(image, axis=1).T
    reconImg = np.absolute(recon)
    return reconImg

lena = plt.imread('lena.png')
r = lena[:,:,0]
g = lena[:,:,1]
b = lena[:,:,2]

pcs = 4
iter = [1, 5, 10, 25, 50, 100, 300, 512]

for i in iter:
    imgR = imageCompression(r,i)
    imgG = imageCompression(g,i)
    imgB = imageCompression(b,i)
    print(i)
    img = lena
    img = np.zeros((512, 512, 3), dtype=float)
    img[:,:,0] = imgR
    img[:,:,1] = imgG
    img[:,:,2] = imgB
    for x in range(512):
        for y in range(512):
            for z in range(3):
                if img[x,y,z] > 1:
                    img[x,y,z] = 1
                elif img[x,y,z] < 0:
                    img[x,y,z] = 0
    plt.imsave("lena" + str(i) + ".png", img)
