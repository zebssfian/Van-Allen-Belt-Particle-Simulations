import numpy as np
import matplotlib.pyplot as plt

# --- important note --- #
'''
The code will run in 3D by default.
    If you wish to view the Earth's magnetic field in 2D,
    you must first comment all the sections marked '3D'
    and uncomment all the sections marked '2D.'

Have fun!
'''

# --- constants --- #
B0 = 3.12e-5 # Tesla (mean value of the magnetic field at the magnetic equator on the Earth's surface)
RE = 6.278 # e6 meters (radius of the Earth)

# --- 3D coordinate system --- #
Dom = np.linspace(-100,100,10)
x, y, z = np.meshgrid(Dom, Dom, Dom)
r = np.sqrt(x**2 + y**2 + z**2)

# --- 2D coordinate system --- #
'''
Dom = np.linspace(-100,100,100)
x, z = np.meshgrid(Dom, Dom)
y = 0
r = np.sqrt(x**2 + y**2 + z**2)
'''

# --- magnetic field --- #
Bx = -3 * B0 * (RE / r)**3 * x*z / r**2
By = -3 * B0 * (RE / r)**3 * y*z / r**2
Bz = -3 * B0 * (RE / r)**3 * (z**2 / r**2 - 1 / 3)

B = np.sqrt(Bx**2+By**2+Bz**2) # field magnitude

# --- Earth as a 3D Ball --- #
u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]
X = RE*np.cos(u)*np.sin(v)
Y = RE*np.sin(u)*np.sin(v)
Z = RE*np.cos(v)

# --- Earth as 2D Disk --- #
'''
circle = plt.Circle((0, 0), RE, color='b',zorder=100)
'''

# --- plotting in 3D --- #
ax = plt.figure().add_subplot(projection='3d')

ax.plot_surface(X, Y, Z, color='b')
ax.quiver(x, y, z, Bx, By, Bz, length = 15, arrow_length_ratio = 0.2, normalize = True)
ax.set_box_aspect([16,16,9])
#ax.set_aspect('auto')
ax.set_facecolor('black')
ax.set_xlim3d([-100, 100])
ax.set_ylim3d([-100, 100])
ax.set_zlim3d([-56, 56])

# make panes transparent
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# remove grid lines
ax.grid(False)

# remove ticks
ax.set_xticks([]) 
ax.set_yticks([]) 
ax.set_zticks([])

# remove tick labels
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# transparent spines
ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

# --- rotating animation --- #
'''
res=10
for angle in range(0, 720*res):
    ax.view_init(15, angle/res)
    plt.draw()
    plt.savefig(str(angle)+".png",dpi=500)
    plt.pause(0.001)
'''

# --- plotting in 2D --- #
'''
fig = plt.figure(figsize = (12, 6))
plt.style.use('dark_background')

plt.streamplot(x, z, Bx, Bz, color=20*np.log(B), density=2)
plt.gca().add_patch(circle)

plt.axis('square')
plt.xlabel('x-axis')
plt.ylabel('z-axis')
'''

plt.show()
