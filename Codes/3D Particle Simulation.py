from mpl_toolkits import mplot3d
import math
import matplotlib.pyplot as plt
import numpy as np

# --- constants --- #
m_t = 1e-14   # kilograms (test mass)
qe = -1.60e-9    # Coulombs (elementary charge) (not even sure this value is correct)
c = 299792458e-6   # meters per second (speed of light)
B0 = 3.12E-5    # Tesla (mean value of the magnetic field at the magnetic equator on the Earth's surface)
RE = 6.278    # meters (radius of the Earth)

# --- numerical integration function --- #
'''
This is our numerical integrator. It goes from 't=0' to 't=tmax,'
    with time steps of size 'eps.' It returns a list of the particle's 
    (x, y, z) coordinates evaluated at every time 't' saved in the list 't_sav.'
'''
def igr(tmax, eps, x0=1.4, y0=0, z0=1.4, xd0=0.01, yd0=0, zd0=0.01, q=qe, m=m_t):    
    # defining initial time t, and defining a list to keep track of t
    t = 0
    t_sav = []

    # defining initial positions x, y, and z; and defining a list to keep track of x, y, and z
    x = x0*RE
    y = y0*RE
    z = z0*RE
    x_sav = []
    y_sav = []
    z_sav = []

    # defining initial velocities xd, yd, and zd; and defining a list to keep track of xd, yd, and zd
    xd = xd0*RE
    yd = yd0*RE
    zd = zd0*RE
    xd_sav = []
    yd_sav = []
    zd_sav = []

    # calculating the radial magnitude, used in computing the magnetic field
    def r(x,y,z):
        return np.sqrt(x**2 + y**2 + z**2)

    # computing Earth's magnetic field strength at a point (x,y,z)
    def Bx(x,y,z):
        return -3 * B0 * (RE / r(x,y,z))**3 * x*z/r(x,y,z)**2
    def By(x,y,z):
        return -3 * B0 * (RE / r(x,y,z))**3 * y*z/r(x,y,z)**2
    def Bz(x,y,z):
        return B0 * (RE / r(x,y,z))**3 * (x**2+y**2-2*z**2) / r(x,y,z)**2

    # --- RK4 numerical integration implementation --- #
    while t <= tmax:
        # time step
        t = t + eps
        t_sav.append(t)

        # to implement RK4, we must split each time interval into 4 steps:
        
        # step 1
        x1 = x
        y1 = y
        z1 = z
        xd1 = xd
        yd1 = yd
        zd1 = zd
        gammainv=np.sqrt(1 - min(1, (xd1**2 + yd1**2 + zd1**2) / c**2)) # inverse Lorentz factor to account for special relativity
        xdd1 = gammainv * (q / m) * (yd1 * Bz(x1,y1,z1) - zd1 * By(x1,y1,z1))
        ydd1 = gammainv * (q / m) * (zd1 * Bx(x1,y1,z1) - xd1 * Bz(x1,y1,z1))
        zdd1 = gammainv * (q / m) * (xd1 * By(x1,y1,z1) - yd1 * Bx(x1,y1,z1))
        
        # step 2
        x2 = x + 0.5 * xd1 * eps
        y2 = y + 0.5 * yd1 * eps
        z2 = z + 0.5 * zd1 * eps
        xd2 = xd + 0.5 * xdd1 * eps
        yd2 = yd + 0.5 * ydd1 * eps
        zd2 = zd + 0.5 * zdd1 * eps
        gammainv = np.sqrt(1 - min(1, (xd2**2 + yd2**2 + zd2**2) / c**2)) # inverse Lorentz factor to account for special relativity
        xdd2 = gammainv * (q / m) * (yd2 * Bz(x2,y2,z2) - zd2 * By(x2,y2,z2))
        ydd2 = gammainv * (q / m) * (zd2 * Bx(x2,y2,z2) - xd2 * Bz(x2,y2,z2))
        zdd2 = gammainv * (q / m) * (xd2 * By(x2,y2,z2) - yd2 * Bx(x2,y2,z2))
        
        # step 3
        x3 = x + 0.5 * xd2 * eps
        y3 = y + 0.5 * yd2 * eps
        z3 = z + 0.5 * zd2 * eps
        xd3 = xd + 0.5 * xdd2 * eps
        yd3 = yd + 0.5 * ydd2 * eps
        zd3 = zd + 0.5 * zdd2 * eps
        gammainv = np.sqrt(1 - min(1, (xd3**2 + yd3**2 + zd3**2) / c**2)) # inverse Lorentz factor to account for special relativity
        xdd3 = gammainv * (q / m) * (yd3 * Bz(x3,y3,z3) - zd3 * By(x3,y3,z3))
        ydd3 = gammainv * (q / m) * (zd3 * Bx(x3,y3,z3) - xd3 * Bz(x3,y3,z3))
        zdd3 = gammainv * (q / m) * (xd3 * By(x3,y3,z3) - yd3 * Bx(x3,y3,z3))
        
        # step 4
        x4 = x + xd3 * eps
        y4 = y + yd3 * eps
        z4 = z + zd3 * eps
        xd4 = xd + xdd3 * eps
        yd4 = yd + ydd3 * eps
        zd4 = zd + zdd3 * eps
        gammainv = np.sqrt(1 - min(1, (xd4**2 + yd4**2 + zd4**2) / c**2)) # inverse Lorentz factor to account for special relativity
        xdd4 = gammainv * (q / m) * (yd4 * Bz(x4,y4,z4) - zd4 * By(x4,y4,z4))
        ydd4 = gammainv * (q / m) * (zd4 * Bx(x4,y4,z4) - xd4 * Bz(x4,y4,z4))
        zdd4 = gammainv * (q / m) * (xd4 * By(x4,y4,z4) - yd4 * Bx(x4,y4,z4))
        
        # final step: averaging the above steps
        x = x + (eps/6) * (xd1 + 2*xd2 + 2*xd3 + xd4)
        y = y + (eps/6) * (yd1 + 2*yd2 + 2*yd3 + yd4)
        z = z + (eps/6) * (zd1 + 2*zd2 + 2*zd3 + zd4)
        xd = xd + (eps/6) * (xdd1 + 2*xdd2 + 2*xdd3 + xdd4)
        yd = yd + (eps/6) * (ydd1 + 2*ydd2 + 2*ydd3 + ydd4)
        zd = zd + (eps/6) * (zdd1 + 2*zdd2 + 2*zdd3 + zdd4)
        
        # saving all the above values for the next time step
        x_sav.append(x)
        y_sav.append(y)
        z_sav.append(z)
        xd_sav.append(xd)
        yd_sav.append(yd)
        zd_sav.append(zd)

    ax.scatter3D(x_sav, y_sav, z_sav, c=t_sav, s = 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')  
    return t, x, xd, y, yd, z, zd

eps = 0.01
tmax = 5000 

ax = plt.axes(projection='3d')

'''
ax.set_xlim3d(0, 300)
ax.set_ylim3d(0, 300)
ax.set_zlim3d(0, 800)
'''

# --- coordinate system --- #
Dom = np.linspace(-5*RE,5*RE,10)
x, y, z = np.meshgrid(Dom, Dom, Dom)
r = np.sqrt(x**2 + y**2 + z**2)

# --- magnetic field --- #
Bx = -3 * B0 * (RE / r)**3 * x*z / r**2
By = -3 * B0 * (RE / r)**3 * y*z / r**2
Bz = -3 * B0 * (RE / r)**3 * (z**2 / r**2 - 1 / 3)

B = np.sqrt(Bx**2 + By**2 + Bz**2) # field magnitude

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
X = RE*np.cos(u)*np.sin(v)
Y = RE*np.sin(u)*np.sin(v)
Z = RE*np.cos(v)
ax.plot_surface(X, Y, Z, color='b')

ax.quiver(x, y, z, Bx, By, Bz, length = 3, arrow_length_ratio = 0.2, normalize = True)

print('final time = ',igr(tmax,eps,x0=2,y0=0,z0=2,xd0=0.01,yd0=0,zd0=0.01,q=qe,m=m_t)[0])
print('final distance = ',igr(tmax,eps,x0=2,y0=0,z0=2,xd0=0.01,yd0=0,zd0=0.01,q=qe,m=m_t)[1])
print('final velocity = ',igr(tmax,eps,x0=2,y0=0,z0=2,xd0=0.01,yd0=0,zd0=0.01,q=qe,m=m_t)[2])

ax.set_xlim3d([-5*RE, 5*RE])
ax.set_ylim3d([-5*RE, 5*RE])
ax.set_zlim3d([-5*RE, 5*RE])
ax.set_box_aspect([16,16,16])
ax.set_facecolor('black')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.tick_params(colors='white')
#ax.set_aspect('auto')

plt.show()
