import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 1.0, 0.01)
s = np.sin(2*np.pi*t)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(t, s, color='blue', lw=2)
fig.show()

print('Lets see whether this works ...')
