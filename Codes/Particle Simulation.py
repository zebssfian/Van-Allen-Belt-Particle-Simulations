# --- required libraries --- #
from mpl_toolkits import mplot3d
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import random
from itertools import count
from IPython import display
import time

# --- constants --- #
m_p=1.67E-27    # kilograms (proton mass)
m_e=9.109E-31   # kilograms (electron mass)
qe=1.602E-19    # Coulombs (elementary charge)
c = 299792458   # meters per second (speed of light)
B0 = 3.12E-5    # Tesla (mean value of the magnetic field at the magnetic equator on the Earth's surface)
RE = 6.278e6    # meters (radius of the Earth)

# --- numerical integration function --- #
'''
This is our numerical integrator. It goes from 't=0' to 't=tmax,'
    with time steps of size 'eps.' It returns a list of the particle's 
    (x, y, z) coordinates evaluated at every time 't' saved in the list 't_sav.'

The code will evaluate the motion of protons by default.
    If you wish to change the particle type,
    you can simply alter the input for 'q' and 'm' below.
'''
def igr(tmax, eps, x0=5, y0=5, z0=5, xd0=1, yd0=1, zd0=1, q=qe, m=m_p):
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
        return -3 * B0 * (RE / r(x,y,z))**3 * x*z / r(x,y,z)**2
    def By(x,y,z):
        return -3 * B0 * (RE / r(x,y,z))**3 * y*z / r(x,y,z)**2
    def Bz(x,y,z):
        return B0 * (RE / r(x,y,z))**3 * (x**2 + y**2 - 2*z**2) / r(x,y,z)**2
    
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
    
    # returning the integration results as lists
    return t_sav, x_sav, y_sav, z_sav

# --- defining the integration step and the final time --- #
eps = 0.01
tmax = 500 # basically in seconds, given the SI units

# --- test cases below --- #
'''
The code will run the first test case by default.
    If you wish to try out other cases,
    you must first comment the default test case
    and uncomment whichever case you want to try out.
    
Feel free to share with us any interesting test cases of your own!
'''

#two particles with proton mass and opposite elementary charge:
t_sav, x_sav, y_sav, z_sav = igr(tmax,eps,x0=5,y0=5,z0=5,xd0=-1,yd0=-1,zd0=-1,q=qe,m=m_p)
t2_sav, x2_sav, y2_sav, z2_sav = igr(tmax,eps,x0=5,y0=5,z0=5,xd0=-1,yd0=-1,zd0=-1,q=-qe,m=m_p)

#two protons symmetric about the z-axis:
'''
t_sav, x_sav, y_sav, z_sav = igr(tmax,eps,x0=5,y0=5,z0=5,xd0=-1,yd0=-1,zd0=-1,q=qe,m=m_p)
t2_sav, x2_sav, y2_sav, z2_sav = igr(tmax,eps,x0=-5,y0=-5,z0=5,xd0=1,yd0=1,zd0=-1,q=qe,m=m_p)
'''

# four protons being trapped in the radiation belt:
'''
t_sav, x_sav, y_sav, z_sav = igr(tmax,eps,x0=10,y0=5,z0=5,xd0=-1,yd0=-1,zd0=-1,q=qe,m=m_p)
t2_sav, x2_sav, y2_sav, z2_sav = igr(tmax,eps,x0=10,y0=5,z0=4,xd0=-1,yd0=-1,zd0=-1,q=qe,m=m_p)
t3_sav, x3_sav, y3_sav, z3_sav = igr(tmax,eps,x0=10,y0=5,z0=3,xd0=-1,yd0=-1,zd0=-1,q=qe,m=m_p)
t4_sav, x4_sav, y4_sav, z4_sav = igr(tmax,eps,x0=10,y0=5,z0=2,xd0=-1,yd0=-1,zd0=-1,q=qe,m=m_p)
'''

# four protons with differing energies:
'''
t_sav, x_sav, y_sav, z_sav = igr(tmax,eps,x0=10,y0=5,z0=5,xd0=-1,yd0=-1,zd0=-1,q=qe,m=m_p)
t2_sav, x2_sav, y2_sav, z2_sav = igr(tmax,eps,x0=10,y0=5,z0=4,xd0=-2,yd0=-1,zd0=-1,q=qe,m=m_p)
t3_sav, x3_sav, y3_sav, z3_sav = igr(tmax,eps,x0=10,y0=5,z0=3,xd0=-3,yd0=-1,zd0=-1,q=qe,m=m_p)
t4_sav, x4_sav, y4_sav, z4_sav = igr(tmax,eps,x0=10,y0=5,z0=2,xd0=-4,yd0=-1,zd0=-1,q=qe,m=m_p)
'''

# four electrons exhibiting relativistic whizzing:
'''
t_sav, x_sav, y_sav, z_sav = igr(tmax,eps,x0=40,y0=5,z0=10,xd0=-1,yd0=-0,zd0=-0.01,q=-qe,m=m_e)
t2_sav, x2_sav, y2_sav, z2_sav = igr(tmax,eps,x0=40,y0=5,z0=5,xd0=-10,yd0=-0,zd0=-0.01,q=-qe,m=m_e)
t3_sav, x3_sav, y3_sav, z3_sav = igr(tmax,eps,x0=40,y0=5,z0=-5,xd0=-1,yd0=-0,zd0=-0.01,q=-qe,m=m_e)
t4_sav, x4_sav, y4_sav, z4_sav = igr(tmax,eps,x0=40,y0=5,z0=-10,xd0=-10,yd0=-0,zd0=-0.01,q=-qe,m=m_e)
'''

# four initially stuck electrons which later escape (requires tmax > 400, v = 40, scal = 5, frames > 200):
'''
t_sav, x_sav, y_sav, z_sav = igr(tmax,eps,x0=40,y0=0,z0=10,xd0=-1,yd0=-0.1,zd0=-1,q=-qe,m=m_e)
t2_sav, x2_sav, y2_sav, z2_sav = igr(tmax,eps,x0=-50,y0=0,z0=5,xd0=1,yd0=-0.1,zd0=-1,q=-qe,m=m_e)
t3_sav, x3_sav, y3_sav, z3_sav = igr(tmax,eps,x0=50,y0=0,z0=-5,xd0=-1,yd0=-0.1,zd0=1,q=-qe,m=m_e)
t4_sav, x4_sav, y4_sav, z4_sav = igr(tmax,eps,x0=-50,y0=0,z0=-10,xd0=1,yd0=-0.1,zd0=1,q=-qe,m=m_e)
'''

# four stuck electrons with nice gyration (requires tmax > 400, v = 1, scal = 4, frames > 500):
'''
t_sav, x_sav, y_sav, z_sav = igr(tmax,eps,x0=40,y0=0,z0=10,xd0=-20,yd0=-1,zd0=-1,q=-qe,m=m_e)
t2_sav, x2_sav, y2_sav, z2_sav = igr(tmax,eps,x0=-50,y0=0,z0=5,xd0=20,yd0=-1,zd0=-1,q=-qe,m=m_e)
t3_sav, x3_sav, y3_sav, z3_sav = igr(tmax,eps,x0=50,y0=0,z0=-5,xd0=-20,yd0=-1,zd0=1,q=-qe,m=m_e)
t4_sav, x4_sav, y4_sav, z4_sav = igr(tmax,eps,x0=-50,y0=0,z0=-10,xd0=20,yd0=-1,zd0=1,q=-qe,m=m_e)
'''

# electron-proton *jingle jingle* tooooo-gether again, badum badum...:
'''
t_sav, x_sav, y_sav, z_sav = igr(tmax,eps,x0=-6,y0=-6,z0=-6,xd0=-1,yd0=-1,zd0=-1,q=qe,m=m_p)
t2_sav, x2_sav, y2_sav, z2_sav = igr(tmax,eps,x0=-50,y0=1,z0=-10,xd0=20,yd0=-5,zd0=1,q=-qe,m=m_e)
'''

# proton leaving the chat:
'''
t_sav, x_sav, y_sav, z_sav = igr(tmax,eps,x0=-5,y0=-5,z0=-5,xd0=5,yd0=5,zd0=5,q=qe,m=m_p)
'''

#proton leaving the chat but more slowly:
'''
t_sav, x_sav, y_sav, z_sav = igr(tmax,eps,x0=-10,y0=-10,z0=-10,xd0=2,yd0=2,zd0=2,q=qe,m=m_p)
'''

#single proton staying:
'''
t_sav, x_sav, y_sav, z_sav = igr(tmax,eps,x0=5,y0=5,z0=5,xd0=1,yd0=1,zd0=1,q=qe,m=m_p)
'''

# string octet:
'''
t_sav, x_sav, y_sav, z_sav = igr(tmax,eps,x0=-35,y0=0,z0=10,xd0=15,yd0=-1,zd0=-1,q=-qe,m=m_e)
t2_sav, x2_sav, y2_sav, z2_sav = igr(tmax,eps,x0=-40,y0=0,z0=5,xd0=20,yd0=-1,zd0=-1,q=-qe,m=m_e)
t3_sav, x3_sav, y3_sav, z3_sav = igr(tmax,eps,x0=-45,y0=0,z0=-5,xd0=25,yd0=-1,zd0=1,q=-qe,m=m_e)
t4_sav, x4_sav, y4_sav, z4_sav = igr(tmax,eps,x0=-50,y0=0,z0=-10,xd0=30,yd0=-1,zd0=1,q=-qe,m=m_e)
t5_sav, x5_sav, y5_sav, z5_sav = igr(tmax,eps,x0=-10,y0=0,z0=4,xd0=1,yd0=-1,zd0=-1,q=qe,m=m_p)
t6_sav, x6_sav, y6_sav, z6_sav = igr(tmax,eps,x0=-10,y0=0,z0=7,xd0=2,yd0=-1,zd0=-1,q=qe,m=m_p)
t7_sav, x7_sav, y7_sav, z7_sav = igr(tmax,eps,x0=-10,y0=0,z0=-7,xd0=3,yd0=-1,zd0=1,q=qe,m=m_p)
t8_sav, x8_sav, y8_sav, z8_sav = igr(tmax,eps,x0=-10,y0=0,z0=-4,xd0=4,yd0=-1,zd0=1,q=qe,m=m_p)
'''

# --- animation --- #
v = 3 # relative animation speed (basically the number of time steps per frame)
scal = 4 # axis scaling factor (can be changed for visuals)

# base aspect ratio setting (approximately 16:9 aspect ratio)
xbase = 100e6
ybase = 56e6

# determines the 2D slice (i.e. xy, xz planes)
slice_2d={12:'', 1:'$x(t)$', 2:'$z(t)$'}

# --- generates Earth's magnetic field in 2D (used as background) --- #
def background():
    # --- 2D coordinate system --- #
    Dom = np.linspace(-scal*xbase, scal*xbase, 800)
    x, z = np.meshgrid(Dom, Dom)
    y = 0 # the axis you set equal to zero depends on your choice of 2D slice, i.e.: (xz-slice => y=0) or (xy-slice => z=0)
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # --- magnetic field --- #
    Bx = -3 * B0 * (RE / r)**3 * x*z / r**2
    By = -3 * B0 * (RE / r)**3 * y*z / r**2
    Bz = -3 * B0 * (RE / r)**3 * (z**2 / r**2 - 1 / 3)

    B = np.sqrt(Bx**2 + Bz**2 + By**2) # field magnitude
    
    # --- Fig Plotting --- #
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    # hello darkness, my old friend #
    plt.style.use('dark_background') # messes with overarching plt settings but it looks better this way
    # we just want a picture since this is a background #
    plt.axis('off')
    # and no wonky earths please #
    plt.gca().axis('square')
    # !!! Earth + field (to change depending on slice) !!! #
    plt.streamplot(x, z, Bx, Bz, color=20*np.log(B), density=2)
    circle = plt.Circle((0, 0), RE, color='#1f77b4',zorder=100)
    plt.gca().add_patch(circle)
    # same limits for consistency #
    ax.set_xlim(xmin=-scal*xbase,xmax=scal*xbase)
    ax.set_ylim(ymin=-scal*ybase,ymax=scal*ybase)
    plt.savefig('dark-Bfield-background.png',bbox_inches='tight',dpi=300,pad_inches=0.0)
    plt.axis('on')  # we don't want plt to be changed forever, only here no axes

# --- run that ^ function once --- #
background()


# --- Animating what we just computed --- #
i=count()                   # the animation frame counter
fig = plt.figure()          # create (and name) a figure
fig.patch.set_facecolor('black')
# for dark mode: #
plt.style.use('dark_background')
# --- create axes --- #
ax = fig.add_subplot(1,1,1) # add and label the relevant figure subplot
# --- get magnetic field background image, will automatically change between xz, xy, yz according to what we did above --- #
background_field = plt.imread('dark-Bfield-background'+slice_2d[12]+'.png')

# --- the animation function (creates the frames) --- #
def animate(j):
    # clear axes for next frame #
    plt.cla()
    # get the current time step #
    t=next(i)
    # --- set the background of the plot --- #
    ax.imshow(background_field, extent=[-scal*xbase,scal*xbase,-scal*ybase,scal*ybase])
    # !!! compute the time-end of the tail of the particle motion !!! #
    tstart = ((t-25)*v if t>25 else 0)
    # plot the particle with its tail, in full color gradient manner #
    ax.scatter(x_sav[tstart:min(t*v,int(tmax*100))],z_sav[tstart:min(t*v,int(tmax*100))],s=4,c=[(col/(t*10)) for col in range(tstart,min(t*v,int(tmax*100)))],marker='o')
    # plot any other particles by copy pasting here #
    ax.scatter(x2_sav[tstart:min(t*v,int(tmax*100))],z2_sav[tstart:min(t*v,int(tmax*100))],s=4,c=[(col/(t*10)) for col in range(tstart,min(t*v,int(tmax*100)))],marker='o')
    #ax.scatter(x3_sav[tstart:min(t*v,int(tmax*100))],z3_sav[tstart:min(t*v,int(tmax*100))],s=4,c=[(col/(t*10)) for col in range(tstart,min(t*v,int(tmax*100)))],marker='o')
    #ax.scatter(x4_sav[tstart:min(t*v,int(tmax*100))],z4_sav[tstart:min(t*v,int(tmax*100))],s=4,c=[(col/(t*10)) for col in range(tstart,min(t*v,int(tmax*100)))],marker='o')
    #ax.scatter(x5_sav[tstart:min(t*v,int(tmax*100))],z5_sav[tstart:min(t*v,int(tmax*100))],s=4,c=[(col/(t*10)) for col in range(tstart,min(t*v,int(tmax*100)))],marker='o')
    #ax.scatter(x6_sav[tstart:min(t*v,int(tmax*100))],z6_sav[tstart:min(t*v,int(tmax*100))],s=4,c=[(col/(t*10)) for col in range(tstart,min(t*v,int(tmax*100)))],marker='o')
    #ax.scatter(x7_sav[tstart:min(t*v,int(tmax*100))],z7_sav[tstart:min(t*v,int(tmax*100))],s=4,c=[(col/(t*10)) for col in range(tstart,min(t*v,int(tmax*100)))],marker='o')
    #ax.scatter(x8_sav[tstart:min(t*v,int(tmax*100))],z8_sav[tstart:min(t*v,int(tmax*100))],s=4,c=[(col/(t*10)) for col in range(tstart,min(t*v,int(tmax*100)))],marker='o')
    
    # !!! change this ^ and the labels below for different slices !!! #
    # label your axes, of course, because we all forget sometimes #
    ax.set_xlabel(slice_2d[1]+str(' at $t = %.2f$' % t_sav[min(t*v,int(tmax*100))])+str(' s')) # note that t is relative, for demonstration purposes of the animation and keeping track of frame count
    ax.set_ylabel(slice_2d[2])
    # and no wonky earths please #
    plt.gca().axis('square')
    # axis limits for consistency #
    ax.set_xlim(xmin=-scal*xbase,xmax=scal*xbase)
    ax.set_ylim(ymin=-scal*ybase,ymax=scal*ybase)

# --- perform the animation using the above function and figure --- #
animation_1 = animation.FuncAnimation(fig,animate,frames=2000,interval=33)
plt.show()
# !!! save the animation !!! #
animation_1.save("particle_motion-proton-electron-together-xz(timed).mp4",dpi=300)


# trace plotting for the full motion #
def plotFullTrace(u,v,t):
    # --- Fig Plotting --- #
    fig = plt.figure()
    plt.style.use('dark_background')
    ax = fig.add_subplot(1,1,1)
    # --- get magnetic field background image, to change between xz, xy, yz --- #
    background_field = plt.imread('dark-Bfield-background'+slice_2d[12]+'.png')
    # --- set the background of the plot --- #
    ax.imshow(background_field, extent=[-scal*xbase,scal*xbase,-scal*ybase,scal*ybase])
    # --- scatter plotter for the particle worldine --- #
    ax.scatter(u,v,s=4,c=[(col/len(t)) for col in range(len(t))],marker='o')
    # consistent axis labelling #
    ax.set_xlabel(slice_2d[1])
    ax.set_ylabel(slice_2d[2])
    plt.gca().axis('square')
    # axis limits for consistency #
    ax.set_xlim(xmin=-scal*xbase,xmax=scal*xbase)
    ax.set_ylim(ymin=-scal*ybase,ymax=scal*ybase)
