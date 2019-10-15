import numpy as np
import matplotlib.pyplot as plt

lena = plt.imread('lena.png')
lena2 = plt.imread('lena512.png')

r = 0
g = 0
b = 0
for x in range(512):
    for y in range(512):
        if lena[x,y,0] != lena2[x,y,0]:
            r += 1
        if lena[x,y,1] != lena2[x,y,1]:
            g += 1
        if lena[x,y,2] != lena2[x,y,2]:
            b += 1
print("Rød:\t" + str(r))
print("Grønn:\t" + str(g))
print("Blå:\t" + str(b))
