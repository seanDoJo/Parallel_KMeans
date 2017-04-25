from PKMeans import PKMeans
from mnist import Numbers
from sklearn.cluster import KMeans
import numpy as np
import time

nums = Numbers()

limit = 5000

data = np.matrix((256*nums.train_x[:limit]).astype(np.float64))
labels = nums.train_y[:limit]
k=9


a, c = PKMeans(data, 100, k=k, iters=150)


for i in range(k):
        m = np.where(a == i)
        print "cluster: {}".format(i)
        local = labels[m]
        total = local.shape[0]
	elems = []
        for j in range(k):
                mm = np.where(local == j)
                perc = mm[0].shape[0] / float(total)
		elems.append((j, perc, mm[0].shape[0]))
	elems.sort(key=lambda x: x[2], reverse=True)

	for l,p,c in elems:
		print "{}: {}, {}".format(l, p, c)
        print ""

# For Debugging
if False:
	print data.shape[0]

	for i in range(k):
		m = np.argmin(c[i,:])
		print "{} {} {} {}".format(c[i,m], i, m, c[i,0:2])
	print np.sum(c[:, c.shape[1]-1])
	print c[:, c.shape[1]-1]
