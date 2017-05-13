import matplotlib.pyplot as plt
import numpy as np

FILENAME = "results_threads.txt"

d = []
t = []

for line in open(FILENAME):
	n, dt, tt = line.split(',')
	d.append(int(dt))
	t.append(float(tt))

d = np.array(d)
t = np.array(t)	

fig, ax = plt.subplots()

ax.plot(d, t, label="Increasing Thread Block Size")
ax.set_xlabel("Closest Kernel Thread Block Size")
ax.set_ylabel("Time (s)")
ax.set_title("Time to Run 100 Iterations vs Thread Block Size")
plt.show()
