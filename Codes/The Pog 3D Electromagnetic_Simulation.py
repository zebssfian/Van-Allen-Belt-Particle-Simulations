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
    
    while t <= tmax:
        t = t + eps
        t_sav.append(t)

        xdd = (q / m) * (yd * Bz(x,y,z) - zd * By(x,y,z)) #* np.sqrt(1 - (xd**2 + yd**2 + zd**2) / c**2)
        ydd = (q / m) * (zd * Bx(x,y,z) - xd * Bz(x,y,z)) #* np.sqrt(1 - (xd**2 + yd**2 + zd**2) / c**2)
        zdd = (q / m) * (xd * By(x,y,z) - yd * Bx(x,y,z)) #* np.sqrt(1 - (xd**2 + yd**2 + zd**2) / c**2)
        
        xd = xd + xdd * eps
        xd_sav.append(xd)

        yd = yd + ydd * eps
        yd_sav.append(yd)
        #print(yd)

        zd = zd + zdd * eps
        zd_sav.append(zd)
        
        x = x + xd * eps
        x_sav.append(x)

        y = y + yd * eps
        y_sav.append(y)

        z = z + zd * eps
        z_sav.append(z)

    ax.scatter3D(x_sav, y_sav, z_sav, c=t_sav)
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
ax.set_box_aspect([16,16,16])
ax.set_aspect('auto')

print('final time = ',igr(tmax,eps,x0=1.4,y0=0,z0=1.4,xd0=0.01,yd0=0,zd0=0.01,q=qe,m=m_t)[0])
print('final distance = ',igr(tmax,eps,x0=1.4,y0=0,z0=1.4,xd0=0.01,yd0=0,zd0=0.01,q=qe,m=m_t)[1])
print('final velocity = ',igr(tmax,eps,x0=1.4,y0=0,z0=1.4,xd0=0.01,yd0=0,zd0=0.01,q=qe,m=m_t)[2])

plt.show()
