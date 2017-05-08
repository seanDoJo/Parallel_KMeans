from EPKMeans import EPKMeans
from mnist import Numbers
import numpy as np
import time

nums = Numbers()

limit = 5000

data = np.matrix((256*nums.train_x[:limit]).astype(np.float64))
labels = nums.train_y[:limit]
k=np.array([11, 9, 7, 5])
#k=np.array([5, 2])


t = time.time()
#a, cl = PKMeans(data, 128, k=k, iters=sk.n_iter_)
a, cl = EPKMeans(data, 128, k=k, iters=150)
print(time.time() - t)

for test in range(k.shape[0]):
        print "Test: {}, k={}".format(test, k[test])
	kk = k[test]
	for i in range(kk):
		m = np.where(a[test,:] == i)
		print "\tcluster: {}".format(i)
		local = labels[m]
		total = local.shape[0]
		if not total:
			continue
		elems = []
		for j in range(kk):
			mm = np.where(local == j)
			perc = mm[0].shape[0] / float(total)
			elems.append((j, perc, mm[0].shape[0]))
		elems.sort(key=lambda x: x[2], reverse=True)

		for l,p,c in elems:
			print "\t\t{}: {}, {}".format(l, p, c)
		print ""

# For Debugging
if False:
	c = cl
	print data.shape[0]

	for i in range(k):
		m = np.argmin(c[i,:])
		print "{} {} {} {}".format(c[i,m], i, m, c[i,0:2])
	print np.sum(c[:, c.shape[1]-1])
	print c[:, c.shape[1]-1]
