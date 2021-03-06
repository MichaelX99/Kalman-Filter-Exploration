#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 23:29:07 2016

@author: mike
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Make the Initial Covariance Matrix
P = 150.0*np.eye(9)


#Make the Dynamic Matrix
dt = 0.01 # Time Step between Filter Steps

A = np.matrix([[1.0, 0.0, 0.0, dt, 0.0, 0.0, 1/2.0*dt**2, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 1/2.0*dt**2, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 1/2.0*dt**2],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

#Make the Measurement Matrix
H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

#Make the Measurement Noise Matrix
rp = 5.0**2  # Noise of Position Measurement
R = np.matrix([[rp, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, rp, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, rp, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, rp, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, rp, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, rp]])


sa = 0.1
G = np.matrix([[1/2.0*dt**2],
               [1/2.0*dt**2],
               [1/2.0*dt**2],
               [dt],
               [dt],
               [dt],
               [1.0],
               [1.0],
               [1.0]])
Q = G*G.T*sa**2


#Make the Disturbance Matrix
B = np.matrix([[0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0]])

#Make the Control Input
u = 0.0

#Make an Identity Matrix
I = np.eye(9)

#Synthetically Create Position Measurement for Ball
Hz = 100.0 # Frequency of Vision System
dt = 1.0/Hz
T = 1.0 # s measuremnt time
m = int(T/dt) # number of measurements

init_pos = 0.0
init_vel = 10.0
init_acc = 10.0

px= 0.0 # x Position Start
py= init_pos # y Position Start
pz= 0.0 # z Position Start

vx = 0.0 # m/s Velocity at the beginning
vy = init_vel # m/s Velocity
vz = 0.0 # m/s Velocity

Xr=[]
Xa=[]
Yr=[]
Ya=[]
Zr=[]
Za=[]
for i in range(int(m)):
    accx = 0
    
    vx += accx*dt
    px += vx*dt

    accz = 0
    vz += accz*dt
    pz += vz*dt
    
    if i < 4:
        accy = init_acc
    else:
        accy = 0
    vy += accy*dt
    py += vy*dt
        
    Xr.append(px)
    Xa.append(accx)
    Yr.append(py)
    Ya.append(accy)
    Zr.append(pz)
    Za.append(accz)

#Add Noise to the position measurement
spr= 1 # Sigma for position noise

Xmr = Xr + spr * (np.random.randn(m))
Ymr = Yr + spr * (np.random.randn(m))
Zmr = Zr + spr * (np.random.randn(m))
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xmr, Ymr, Zmr, c='gray')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Ball Trajectory observed from Computer Vision System (with Noise)')

# Axis equal
max_range = np.array([Xmr.max()-Xmr.min(), Ymr.max()-Ymr.min(), Zmr.max()-Zmr.min()]).max() / 3.0
mean_x = Xmr.mean()
mean_y = Ymr.mean()
mean_z = Zmr.mean()
ax.set_xlim(mean_x - max_range, mean_x + max_range)
ax.set_ylim(mean_y - max_range, mean_y + max_range)
ax.set_zlim(mean_z - max_range, mean_z + max_range)

#Add Noise to the acceleration measurement
spa= .1 # Sigma for position noise

Xma = Xa + spa * (np.random.randn(m))
Yma = Ya + spa * (np.random.randn(m))
Zma = Za + spa * (np.random.randn(m))
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xma, Yma, Zma, c='gray')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Ball Acceleration observed from Accelerometer (with Noise)')

# Axis equal
max_range = np.array([Xma.max()-Xma.min(), Yma.max()-Yma.min(), Zma.max()-Zma.min()]).max() / 3.0
mean_x = Xma.mean()
mean_y = Yma.mean()
mean_z = Zma.mean()
ax.set_xlim(mean_x - max_range, mean_x + max_range)
ax.set_ylim(mean_y - max_range, mean_y + max_range)
ax.set_zlim(mean_z - max_range, mean_z + max_range)

measurements = np.vstack((Xmr,Ymr,Zmr,Xma,Yma,Zma))

#Make the initial state
x = np.matrix([0.0, init_pos, 0.0, 0.0, init_vel, 0.0, 0.0, init_acc, 0.0]).T
              
#Preallocation for plotting
xt = []
yt = []
zt = []
dxt= []
dyt= []
dzt= []
ddxt=[]
ddyt=[]
ddzt=[]
Zx = []
Zy = []
Zz = []
Px = []
Py = []
Pz = []
Pdx= []
Pdy= []
Pdz= []
Pddx=[]
Pddy=[]
Pddz=[]
Kx = []
Ky = []
Kz = []
Kdx= []
Kdy= []
Kdz= []
Kddx=[]
Kddy=[]
Kddz=[]

#Run the Kalman Filter
for filterstep in range(m):    
    # Time Update (Prediction)
    # ========================
    # Project the state ahead
    x = A*x + B*u
    
    # Project the error covariance ahead
    P = A*P*A.T + Q    
    
    
    # Measurement Update (Correction)
    # ===============================
    # Compute the Kalman Gain
    S = H*P*H.T + R
    K = (P*H.T) * np.linalg.pinv(S)

    
    # Update the estimate via z
    Z = measurements[:,filterstep].reshape(H.shape[0],1)
    y = Z - (H*x)                            # Innovation or Residual
    x = x + (K*y)
    
    # Update the error covariance
    P = (I - (K*H))*P
    
   
    
    # Save states for Plotting
    xt.append(float(x[0]))
    yt.append(float(x[1]))
    zt.append(float(x[2]))
    dxt.append(float(x[3]))
    dyt.append(float(x[4]))
    dzt.append(float(x[5]))
    ddxt.append(float(x[6]))
    ddyt.append(float(x[7]))
    ddzt.append(float(x[8]))
    
    Zx.append(float(Z[0]))
    Zy.append(float(Z[1]))
    Zz.append(float(Z[2]))
    Px.append(float(P[0,0]))
    Py.append(float(P[1,1]))
    Pz.append(float(P[2,2]))
    Pdx.append(float(P[3,3]))
    Pdy.append(float(P[4,4]))
    Pdz.append(float(P[5,5]))
    Pddx.append(float(P[6,6]))
    Pddy.append(float(P[7,7]))
    Pddz.append(float(P[8,8]))
    Kx.append(float(K[0,0]))
    Ky.append(float(K[1,0]))
    Kz.append(float(K[2,0]))
    Kdx.append(float(K[3,0]))
    Kdy.append(float(K[4,0]))
    Kdz.append(float(K[5,0]))
    Kddx.append(float(K[6,0]))
    Kddy.append(float(K[7,0]))
    Kddz.append(float(K[8,0]))

#Show overlayed position
fig = plt.figure(figsize=(16,9))

plt.plot(xt,zt, label='Kalman Filter Estimate')
plt.scatter(Xmr,Zmr, label='Measurement', c='gray', s=30)
plt.plot(Xr, Zr, label='Real')
plt.title('Estimate of Ball Trajectory (Elements from State Vector $x$)')
plt.legend(loc='best',prop={'size':22})
plt.axhline(0, color='k')
plt.axis('equal')
plt.xlabel('X ($m$)')
plt.ylabel('Y ($m$)')
plt.ylim(0, 2);
plt.savefig('Kalman-Filter-CA-Ball-StateEstimated.png', dpi=150, bbox_inches='tight')

fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111, projection='3d')
ax.plot(xt,yt,zt, label='Kalman Filter Estimate')
ax.plot(Xr, Yr, Zr, label='Real')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('Ball Trajectory estimated with Kalman Filter')

# Axis equal
max_range = np.array([Xmr.max()-Xmr.min(), Ymr.max()-Ymr.min(), Zmr.max()-Zmr.min()]).max() / 3.0
mean_x = Xmr.mean()
mean_y = Ymr.mean()
mean_z = Zmr.mean()
ax.set_xlim(mean_x - max_range, mean_x + max_range)
ax.set_ylim(mean_y - max_range, mean_y + max_range)
ax.set_zlim(mean_z - max_range, mean_z + max_range)
plt.savefig('Kalman-Filter-CA-Ball-Trajectory.png', dpi=150, bbox_inches='tight')

#Say how far away we actually are
dist = np.sqrt((Xmr-xt)**2 + (Ymr-yt)**2 + (Zmr-zt)**2)
print('Estimated Position is %.2fm away from ball position.' % dist[-1])