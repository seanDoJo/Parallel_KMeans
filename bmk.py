import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PKMeans import PKMeans
from KMeans import Kmeans

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
KK = 5


sk = KMeans(init='k-means++', n_clusters=KK, n_init=1)

xs = np.array([])
ys = np.array([])
zs = np.array([])
for i in range(KK):
	# How many points to generate per cluster
        xs1, ys1, zs1 = generate(10000, (i+1)*10, 5)
        xs = np.concatenate((xs, xs1))
        ys = np.concatenate((ys, ys1))
        zs = np.concatenate((zs, zs1))

d = np.matrix([xs, ys, zs]).transpose()

s = time.time()
sk.fit(d)
print(time.time() - s)

start = time.time()
#ah,c = PKMeans(d, 100, k=KK, iters=150)
ah,c = PKMeans(d, 100, k=KK, iters=150, shared=True)
print(time.time() - start)


#start = time.time()
#ahh, it = Kmeans(KK, d)
#print(time.time() - start)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax1 = fig.add_subplot(122, projection='3d')
colors = get_color()

for i in range(KK):
    c = d[np.where(ah == float(i))[0], :]
    col = next(colors)
    print col

    xss = np.squeeze(np.asarray(c[:,0]))
    yss = np.squeeze(np.asarray(c[:,1]))
    zss = np.squeeze(np.asarray(c[:,2]))

    ax.scatter(xss, yss, zss, c=col)
plt.show()
"""
for i in range(KK):
    c = d[np.where(ahh == float(i))[0], :]
    col = next(colors)

    xss = np.squeeze(np.asarray(c[:,0]))
    yss = np.squeeze(np.asarray(c[:,1]))
    zss = np.squeeze(np.asarray(c[:,2]))

    ax1.scatter(xss, yss, zss, c=col)
plt.show()
"""
