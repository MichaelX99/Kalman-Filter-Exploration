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
P_both = 150.0*np.eye(9)
P_gps = 150.0*np.eye(9)

#Make the Dynamic Matrix
dt = 0.01 # Time Step between Filter Steps

A_both = np.matrix([[1.0, 0.0, 0.0, dt, 0.0, 0.0, 1/2.0*dt**2, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 1/2.0*dt**2, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 1/2.0*dt**2],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
A_gps = np.matrix([[1.0, 0.0, 0.0, dt, 0.0, 0.0, 1/2.0*dt**2, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 1/2.0*dt**2, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 1/2.0*dt**2],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])


#Make the Measurement Matrix
H_both = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
H_gps = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


#Make the Measurement Noise Matrix
rp = 5.0**2  # Noise of Position Measurement
R_both = np.matrix([[rp, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, rp, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, rp, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, rp, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, rp, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, rp]])
R_gps = np.matrix([[rp, 0.0, 0.0],
               [0.0, rp, 0.0],
               [0.0, 0.0, rp]])



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
Q_both = G*G.T*sa**2
Q_gps = G*G.T*sa**2

#Make the Disturbance Matrix
B_both = np.matrix([[0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0]])
B_gps = np.matrix([[0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0]])


#Make the Control Input
u_both = 0.0
u_gps = 0.0

#Make an Identity Matrix
I_both = np.eye(9)
I_gps = np.eye(9)

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

measurements_both = np.vstack((Xmr,Ymr,Zmr,Xma,Yma,Zma))
measurements_gps = np.vstack((Xmr,Ymr,Zmr))

#Make the initial state
x_both = np.matrix([0.0, init_pos, 0.0, 0.0, init_vel, 0.0, 0.0, init_acc, 0.0]).T
x_gps = np.matrix([0.0, init_pos, 0.0, 0.0, init_vel, 0.0, 0.0, init_acc, 0.0]).T
              
#Preallocation for plotting
xt_both = []
yt_both = []
zt_both = []
xt_gps = []
yt_gps = []
zt_gps = []


#Run the Kalman Filter
for filterstep in range(m):    
    # Time Update (Prediction)
    # ========================
    # Project the state ahead
    x_both = A_both*x_both + B_both*u_both
    x_gps = A_gps*x_gps + B_gps*u_gps
    
    # Project the error covariance ahead
    P_both = A_both*P_both*A_both.T + Q_both    
    P_gps = A_gps*P_gps*A_gps.T + Q_gps
    
    # Measurement Update (Correction)
    # ===============================
    # Compute the Kalman Gain
    S_both = H_both*P_both*H_both.T + R_both
    K_both = (P_both*H_both.T) * np.linalg.pinv(S_both)
    S_gps = H_gps*P_gps*H_gps.T + R_gps
    K_gps = (P_gps*H_gps.T) * np.linalg.pinv(S_gps)

    
    # Update the estimate via z
    Z_both = measurements_both[:,filterstep].reshape(H_both.shape[0],1)
    y_both = Z_both - (H_both*x_both)                            # Innovation or Residual
    x_both = x_both + (K_both*y_both)
    Z_gps = measurements_gps[:,filterstep].reshape(H_gps.shape[0],1)
    y_gps = Z_gps - (H_gps*x_gps)                            # Innovation or Residual
    x_gps = x_gps + (K_gps*y_gps)
    
    # Update the error covariance
    P_both = (I_both - (K_both*H_both))*P_both
    P_gps = (I_gps - (K_gps*H_gps))*P_gps
   
    
    # Save states for Plotting
    xt_both.append(float(x_both[0]))
    yt_both.append(float(x_both[1]))
    zt_both.append(float(x_both[2]))
    xt_gps.append(float(x_gps[0]))
    yt_gps.append(float(x_gps[1]))
    zt_gps.append(float(x_gps[2]))
    
#Show overlayed position
fig = plt.figure(figsize=(16,9))

plt.plot(xt_both,zt_both, label='GPS and Accelerometer Kalman Filter Estimate')
plt.scatter(Xmr,Zmr, label='GPS Measurement', c='gray', s=30)
plt.plot(xt_gps,zt_gps, label='GPS Kalman Filter Estimate')
plt.title('Estimate of Ball Trajectory (Elements from State Vector $x$)')
plt.legend(loc='best',prop={'size':22})
plt.axhline(0, color='k')
plt.axis('equal')
plt.xlabel('X ($m$)')
plt.ylabel('Y ($m$)')
plt.ylim(0, 2);

fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111, projection='3d')
ax.plot(xt_both,yt_both,zt_both, label='GPS and Accelerometer Kalman Filter Estimate')
ax.plot(xt_gps,yt_gps,zt_gps, label='GPS Kalman Filter Estimate')
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

#Say how far away we actually are
dist_both = np.sqrt((Xmr-xt_both)**2 + (Ymr-yt_both)**2 + (Zmr-zt_both)**2)
dist_gps = np.sqrt((Xmr-xt_gps)**2 + (Ymr-yt_gps)**2 + (Zmr-zt_gps)**2)
print('Estimated Position of GPS and Accelerometer Kalman Filter is %.2fm away from ball position.' % dist_both[-1])
print('Estimated Position of GPS Kalman Filter is %.2fm away from ball position.' % dist_gps[-1])