import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from EPKMeans import EPKMeans

from sklearn.cluster import KMeans

import time

def generate(numpoints, ctr, std):
    centers = np.random.uniform(0, ctr, size=3)
    stdevs = np.random.uniform(1, std, size=3)
    xs = np.random.normal(centers[0], stdevs[0], (1, numpoints))
    ys = np.random.normal(centers[1], stdevs[1], (1, numpoints))
    zs = np.random.normal(centers[2], stdevs[2], (1, numpoints))

    return (xs[0], ys[0], zs[0])

def get_color():
    colors = ['g', 'c', 'y', 'b', 'r', 'm', 'k']
    i = 0
    while True:
        yield colors[i]
        i = (i + 1) % 7


# Number of clusters
KK = np.array([2, 3, 4, 5])


xs = np.array([])
ys = np.array([])
zs = np.array([])
for i in range(5):
	# How many points to generate per cluster
        xs1, ys1, zs1 = generate(1000, (i+1)*20, 5)
        xs = np.concatenate((xs, xs1))
        ys = np.concatenate((ys, ys1))
        zs = np.concatenate((zs, zs1))

d = np.matrix([xs, ys, zs]).transpose()

start = time.time()
ah, c = EPKMeans(d, 100, k=KK, iters=150)
print(time.time() - start)



fig = plt.figure()

ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(223, projection='3d')
ax4 = fig.add_subplot(224, projection='3d')
colors = get_color()

for i in range(2):
    c = d[np.where(ah[0,:] == float(i))[0], :]
    col = next(colors)

    xss = np.squeeze(np.asarray(c[:,0]))
    yss = np.squeeze(np.asarray(c[:,1]))
    zss = np.squeeze(np.asarray(c[:,2]))

    ax1.scatter(xss, yss, zss, c=col)

for i in range(3):
    c = d[np.where(ah[1,:] == float(i))[0], :]
    col = next(colors)

    xss = np.squeeze(np.asarray(c[:,0]))
    yss = np.squeeze(np.asarray(c[:,1]))
    zss = np.squeeze(np.asarray(c[:,2]))

    ax2.scatter(xss, yss, zss, c=col)

for i in range(4):
    c = d[np.where(ah[2,:] == float(i))[0], :]
    col = next(colors)

    xss = np.squeeze(np.asarray(c[:,0]))
    yss = np.squeeze(np.asarray(c[:,1]))
    zss = np.squeeze(np.asarray(c[:,2]))

    ax3.scatter(xss, yss, zss, c=col)

for i in range(5):
    c = d[np.where(ah[3,:] == float(i))[0], :]
    col = next(colors)

    xss = np.squeeze(np.asarray(c[:,0]))
    yss = np.squeeze(np.asarray(c[:,1]))
    zss = np.squeeze(np.asarray(c[:,2]))

    ax4.scatter(xss, yss, zss, c=col)

plt.show()
