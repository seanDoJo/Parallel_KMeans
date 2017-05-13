import matplotlib.pyplot as plt
import numpy as np

FILENAME = "results_breakdown.txt"

d = {"bmk": [], "mnist": []}
closest = {"bmk": [], "mnist": []}
zero = {"bmk": [], "mnist": []}
sm = {"bmk": [], "mnist": []}
avg = {"bmk": [], "mnist": []}
transfer = {"bmk": [], "mnist": []}
total = {"bmk": [], "mnist": []}

for line in open(FILENAME):
	n, dt, c, z, s, a, tr, tt = line.split(',')
	d[n].append(int(dt))
	closest[n].append(float(c))
	zero[n].append(float(z))
	sm[n].append(float(s))
	avg[n].append(float(a))
	transfer[n].append(float(tr))
	total[n].append(float(tt))


fig, ax = plt.subplots()

ax.plot(d["mnist"], closest["mnist"], label="Closest Kernel")
ax.plot(d["mnist"], zero["mnist"], label="Zero Kernel")
ax.plot(d["mnist"], sm["mnist"], label="Sum Kernel")
ax.plot(d["mnist"], avg["mnist"], label="Average Kernel")
ax.plot(d["mnist"], transfer["mnist"], label="Transfer Overhead")

ax.set_xlabel("Number of Samples")
ax.set_ylabel("Time (s)")
ax.set_title("Kernel Part Execution Times (MNIST Dataset)")

legend = ax.legend(loc="upper left")

fig1, ax1 = plt.subplots()

ax1.plot(d["bmk"], closest["bmk"], label="Closest Kernel")
ax1.plot(d["bmk"], zero["bmk"], label="Zero Kernel")
ax1.plot(d["bmk"], sm["bmk"], label="Sum Kernel")
ax1.plot(d["bmk"], avg["bmk"], label="Average Kernel")
ax1.plot(d["bmk"], transfer["bmk"], label="Transfer Overhead")

ax1.set_xlabel("Number of Samples")
ax1.set_ylabel("Time (s)")
ax1.set_title("Kernel Part Execution Times (3D Dataset)")

legend = ax1.legend(loc="upper left")

fig2, ax2 = plt.subplots()

ax2.plot(d["bmk"], closest["bmk"], label="Closest Kernel (3D)")
ax2.plot(d["mnist"], closest["mnist"], label="Closest Kernel (MNIST)")

ax2.set_xlabel("Number of Samples")
ax2.set_ylabel("Time (s)")
ax2.set_title("Closest Kernel Execution Times")

legend = ax2.legend(loc="upper left")

fig3, ax3 = plt.subplots()

ax3.plot(d["bmk"], sm["bmk"], label="Sum Kernel (3D)")
ax3.plot(d["mnist"], sm["mnist"], label="Sum Kernel (MNIST)")

ax3.set_xlabel("Number of Samples")
ax3.set_ylabel("Time (s)")
ax3.set_title("Sum Kernel Execution Times")

legend = ax3.legend(loc="upper left")

plt.show()
