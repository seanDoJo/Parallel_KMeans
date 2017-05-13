import matplotlib.pyplot as plt
import numpy as np

FILENAME = "results_data.txt"

d = {"bmk": [], "mnist": []}
t = {"bmk": [], "mnist": []}

for line in open(FILENAME):
	n, dt, tt = line.split(',')
	d[n].append(int(dt))
	t[n].append(float(tt))


fig, ax = plt.subplots()

ax.plot(d["mnist"], t["mnist"], label="MNIST Dataset")
ax.plot(d["bmk"], t["bmk"], label="3D Dataset")

ax.set_xlabel("Number of Samples")
ax.set_ylabel("Time (s)")
ax.set_title("Time to Run 100 Iterations vs Number of Samples")

legend = ax.legend(loc="lower right")

plt.show()
