import matplotlib.pyplot as plt
import numpy as np

FILENAME = "results_speedup.txt"

d = []
sk = []
pk = []
spe = []

for line in open(FILENAME):
	dt, stt, ptt, sp = line.split(',')
	d.append(int(dt))
	sk.append(float(stt))
	pk.append(float(ptt))
	spe.append(float(sp))


fig, ax = plt.subplots()

ax.plot(d, sk, label="SKLearn Implementation")
ax.plot(d, pk, label="PKMeans Implementation")

ax.set_xlabel("Number of Samples")
ax.set_ylabel("Time (s)")
ax.set_title("Execution Times for SKLearn and PKMeans")
ax.legend(loc="upper left")

fig1, ax1 = plt.subplots()
ax1.plot(d, spe, label="Speedup")
ax1.set_xlabel("Number of Samples")
ax1.set_ylabel("Relative Speedup")
ax1.set_title("Relative Speedup of PKMeans Over SKLearn")
plt.show()
