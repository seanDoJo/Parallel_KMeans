import matplotlib.pyplot as plt
import numpy as np

FILENAME = "results_dimensionality.txt"

d = []
t = []

for line in open(FILENAME):
	dt, tt = line.split(',')
	d.append(int(dt))
	t.append(float(tt))

d = np.array(d)
t = np.array(t)	

fig, ax = plt.subplots()

ax.plot(d, t, label="Increasing Dimensionality")
ax.set_xlabel("Dimensionality")
ax.set_ylabel("Time (s)")
ax.set_title("Time to Run 100 Iterations vs Dimensionality of Data")
plt.show()
