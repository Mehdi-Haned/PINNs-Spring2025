
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import ellipj

# Create the figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)

# Initial parameters
u = np.arange(0, 10, 1e-2)
initial_k = 0
m = initial_k**2

sn, cn, dn, am = ellipj(u, m)
[line1] = ax.plot(u, sn, label='sn(u, k)')
[line2] = ax.plot(u, cn, label='cn(u, k)')
[line3] = ax.plot(u, dn, label='dn(u, k)')
[line4] = ax.plot(u, am, label='am(u, k)')
ax.set_title("Jacobi Elliptic Function sn(u, k)")
ax.set_xlabel("u")
ax.set_ylabel("sn(u, k)")
ax.set_ylim(-1.1, 5.1)
ax.legend()

# Slider axis and slider
ax_k = plt.axes([0.1, 0.1, 0.8, 0.05])
slider_k = Slider(ax_k, 'k', 0.0, 0.99, valinit=initial_k, valstep=0.01)

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
