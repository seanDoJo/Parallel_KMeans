import matplotlib.pyplot as plt
import numpy as np

FILENAME = "results_shared.txt"

d = []
shared = []
unshared = []
spe = []

for line in open(FILENAME):
	n, dt, sh, un, sp = line.split(',')
	d.append(int(dt))
	shared.append(float(sh))
	unshared.append(float(un))
	spe.append(float(sp))


fig, ax = plt.subplots()

ax.plot(d, shared, label="Shared Memory")
ax.plot(d, unshared, label="Global Memory")

ax.set_xlabel("Number of Samples")
ax.set_ylabel("Time (s)")
ax.set_title("Execution Times for PKMeans With and Without Shared Memory")
ax.legend(loc="upper left")

fig1, ax1 = plt.subplots()
ax1.plot(d, spe, label="Speedup")
ax1.set_xlabel("Number of Samples")
ax1.set_ylabel("Relative Speedup")
ax1.set_title("Relative Speedup of Shared Memory Over Global Memory")
plt.show()
