import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import ellipj

# Create the figure and axis
fig, axs = plt.subplots(nrows=2, ncols=2)
plt.subplots_adjust(left=0.1, bottom=0.25)

# Initial parameters
u         = np.arange(0, 20, 1e-2)
initial_k = 0
m         = initial_k**2

sn, cn, dn, am = ellipj(u, m)
[line1] = axs[0,0].plot(u, sn, label='sn(u, k)', color="blue")
[line2] = axs[1,0].plot(u, cn, label='cn(u, k)', color="green")
[line4] = axs[0,1].plot(u, am, label='am(u, k)', color="red")
[line3] = axs[1,1].plot(u, dn, label='dn(u, k)', color="black")


titles  = ['sn(u, k)', 'am(u, k)', 'cn(u, k)', 'dn(u, k)']
ylimits = [(-1.1,1.1), (u[0], u[-1]*1.1), (-1.1,1.1), (-0.5,2.5)]

# Flatten the 2D array of axes and assign titles
for ax, title, yl in zip(axs.flat, titles, ylimits):
    ax.set_title(title)
    ax.grid()
    ax.set_ylim(yl[0], yl[1])

# Slider axis and slider
ax_k     = plt.axes([0.1, 0.1, 0.8, 0.05])
slider_k = Slider(ax_k, 'k', -1, 1, valinit=initial_k, valstep=1e-2)

# Update function
def update(val):
    k = slider_k.val
    m = k**2
    sn, cn, dn, am = ellipj(u, m)
    line1.set_ydata(sn)
    line2.set_ydata(cn)
    line3.set_ydata(dn)
    line4.set_ydata(am)
    fig.canvas.draw_idle()

slider_k.on_changed(update)

plt.show()
