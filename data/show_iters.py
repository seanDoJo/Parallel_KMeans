import matplotlib.pyplot as plt
import numpy as np

FILENAME = "results_iters.txt"

d = {"bmk": [], "mnist": []}
t = {"bmk": [], "mnist": []}

for line in open(FILENAME):
	n, dt, tt = line.split(',')
	d[n].append(int(dt))
	t[n].append(float(tt))


fig, ax = plt.subplots()

ax.plot(d["mnist"], t["mnist"], label="MNIST Dataset")
ax.plot(d["bmk"], t["bmk"], label="3D Dataset")

ax.set_xlabel("Number of Iterations")
ax.set_ylabel("Time (s)")
ax.set_title("Time to Execute vs Number of Iterations")

legend = ax.legend(loc="lower right")

plt.show()
