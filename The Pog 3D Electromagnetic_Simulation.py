from mpl_toolkits import mplot3d
import math
import matplotlib.pyplot as plt
import numpy as np

q = -1.60e-9 # C
m = 1e-14 # kg
c = 299792458e-6 # e6 m/s
B0 = 3.12e-5 # T
RE = 6.278 # e6 m
    
def igr(eps):
    x0 = 1.4*RE
    y0 = 0
    z0 = 1.4*RE
    
    t = 0
    t_sav = []

    x = x0
    y = y0
    z = z0
    x_sav = []
    y_sav = []
    z_sav = []

    xd = 0.01 * RE
    yd = 0 * RE
    zd = 0.01*RE
    xd_sav = []
    yd_sav = []
    zd_sav = []

    def r(x,y,z):
        return np.sqrt(x**2 + y**2 + z**2)
    
    def Bx(x,y,z):
        return -3 * B0 * (RE / r(x,y,z))**3 * x*z/r(x,y,z)**2
    def By(x,y,z):
        return -3 * B0 * (RE / r(x,y,z))**3 * y*z/r(x,y,z)**2
    def Bz(x,y,z):
        return B0 * (RE / r(x,y,z))**3 * (x**2+y**2-2*z**2) / r(x,y,z)**2
    #print(Bx, By, Bz)
    
    while t <= 1000:
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

ax = plt.axes(projection='3d')

'''
ax.set_xlim3d(0, 300)
ax.set_ylim3d(0, 300)
ax.set_zlim3d(0, 800)
'''

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
X = RE*np.cos(u)*np.sin(v)
Y = RE*np.sin(u)*np.sin(v)
Z = RE*np.cos(v)
ax.plot_surface(X, Y, Z)
ax.set_aspect('equal')

print('final time = ',igr(eps)[0])
print('final distance = ',igr(eps)[1])
print('final velocity = ',igr(eps)[2])

plt.show()
