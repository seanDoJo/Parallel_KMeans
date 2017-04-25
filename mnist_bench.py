from PKMeans import PKMeans
from mnist import Numbers
from sklearn.cluster import KMeans
import numpy as np
import time

nums = Numbers()

datax = np.matrix((256*nums.train_x).astype(np.float64))
labels = nums.train_y
k=9

amnts = [500, 1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]

tpb = [32, 64, 128, 256, 512]

for t in tpb:
	for i in amnts:

		data = datax[:i,:]

		sk = KMeans(init='random', n_clusters=k, n_init=10)

		s = time.time()
		sk.fit(data)
		e = time.time()

		start = time.time()
		a, c = PKMeans(data, 100, k=k, iters=sk.n_iter_, tpb=t)
		end = time.time()

		tk = (e-s)
		tp = (end - start)

		print "{},{},{},{}".format(t, i, (tk / tp), sk.n_iter_)

print ""

for t in tpb:
	for i in amnts:

		data = datax[:i,:]

		start = time.time()
		a, c = PKMeans(data, 100, k=k, iters=300, tpb=t)
		end = time.time()

		print "{},{},{}".format(t,i, (end - start))

exit(0)

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

print(end - start)


if False:
	print data.shape[0]

	for i in range(k):
		m = np.argmin(c[i,:])
		print "{} {} {} {}".format(c[i,m], i, m, c[i,0:2])
	print np.sum(c[:, c.shape[1]-1])
	print c[:, c.shape[1]-1]
