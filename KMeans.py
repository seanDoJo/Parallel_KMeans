import numpy as np
import matplotlib.pyplot as plt
import math
import time

def get_color():
    colors = ['g', 'c', 'y', 'b', 'r', 'm', 'k']
    i = 0
    while True:
        yield colors[i]
        i = (i + 1) % 7

def Kmeans(k, data):
    oldkv = np.zeros((k, data.shape[1]))
    centroid = np.random.uniform(-100, 100, size=(k, data.shape[1]))
    centroidl = np.array([np.random.uniform(-100, 100, size=(k, data.shape[1])), np.zeros((k, data.shape[1]))])
    curr = 1
    old = 0
    cent_dists = np.zeros((data.shape[0], k))
    cent_assigns = np.zeros(data.shape[0])

    iters = 0
    while np.sum((centroidl[curr] - centroidl[old])):
	iters += 1
        for j in range(k):
            shifted = data - centroidl[old][j,:]
            squared = np.multiply(shifted, shifted)
            summed = np.dot(squared, np.ones((data.shape[1], 1)))
            cent_dists[:,j] = np.squeeze(np.asarray(summed))
        for j in range(data.shape[0]):
            cent_assigns[j] = np.argmin(cent_dists[j,:])
        for j in range(k):
            c = data[np.where(cent_assigns == float(j))[0], :]
            if not c.shape[0]:
                centroidl[curr][j,:] = np.random.uniform(-100, 100, size=(1, 3))
                continue
            centroidl[curr][j,:] = np.squeeze(np.asarray(np.sum(c, axis=0) / c.shape[0]))

        curr, old = old, curr
    return (cent_assigns, iters)

def generate(numpoints, ctr, std):
    centers = np.random.uniform(-ctr, ctr, size=2)
    stdevs = np.random.uniform(1, std, size=2)
    xs = np.random.normal(centers[0], stdevs[0], (1, numpoints))
    ys = np.random.normal(centers[1], stdevs[1], (1, numpoints))


    return (xs[0], ys[0])

def generate3(numpoints, ctr, std):
    centers = np.random.uniform(0, ctr, size=3)
    stdevs = np.random.uniform(1, std, size=3)
    xs = np.random.normal(centers[0], stdevs[0], (1, numpoints))
    ys = np.random.normal(centers[1], stdevs[1], (1, numpoints))
    zs = np.random.normal(centers[2], stdevs[2], (1, numpoints))


    return (xs[0], ys[0], zs[0])

"""
xs = np.array([])
ys = np.array([])
for i in range(4):
    xs1, ys1 = generate(100, i*100, 1000)
    xs = np.concatenate((xs, xs1))
    ys = np.concatenate((ys, ys1))

d = np.matrix([xs, ys]).transpose()

xs = np.array([])
ys = np.array([])
zs = np.array([])
for i in range(10):
    xs1, ys1, zs1 = generate3(5000, (i+1)*20, 5)
    xs = np.concatenate((xs, xs1))
    ys = np.concatenate((ys, ys1))
    zs = np.concatenate((zs, zs1))

d = np.matrix([xs, ys, zs]).transpose()

start = time.time()
cent_assigns = Kmeans(10, d)
print(time.time() - start)
exit(0)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = get_color()
for i in range(7):
    c = d[np.where(cent_assigns == float(i))[0], :]
    col = next(colors)

    xs = np.squeeze(np.asarray(c[:,0]))
    ys = np.squeeze(np.asarray(c[:,1]))
    zs = np.squeeze(np.asarray(c[:,2]))
    
    ax.scatter(xs, ys, zs, c=col)
plt.show()
"""
