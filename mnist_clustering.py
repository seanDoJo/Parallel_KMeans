from PKMeans import PKMeans
from mnist import Numbers
from sklearn.cluster import KMeans
import numpy as np
import time

nums = Numbers()

limit = 50000

data = np.matrix((256*nums.train_x[:limit]).astype(np.float64))
labels = nums.train_y[:limit]
k=9

sk = KMeans(init='random', n_clusters=k, n_init=1)

t = time.time()
sk.fit(data)
print(time.time() - t)

print(sk.n_iter_)

t = time.time()
a, cl = PKMeans(data, 128, k=k, iters=sk.n_iter_)
#a, cl = PKMeans(data, 128, k=k, iters=150)
print(time.time() - t)

t = time.time()
aa, cll = PKMeans(data, 128, k=k, iters=sk.n_iter_, shared=False)
#a, cl = PKMeans(data, 128, k=k, iters=150)
print(time.time() - t)
exit(0)

for i in range(k):
        m = np.where(a == i)
        print "cluster: {}".format(i)
        local = labels[m]
        total = local.shape[0]
	if not total:
		continue
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
if True:
	c = cl
	print data.shape[0]

	for i in range(k):
		m = np.argmin(c[i,:])
		print "{} {} {} {}".format(c[i,m], i, m, c[i,0:2])
	print np.sum(c[:, c.shape[1]-1])
	print c[:, c.shape[1]-1]
