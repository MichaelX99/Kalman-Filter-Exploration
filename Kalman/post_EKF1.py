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
P = 15.0*np.eye(9)

#Make the Dynamic Matrix
dt = 1/100 # Time Step between Filter Steps

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
T = 1 # s measuremnt time
m = int(T/dt) # number of measurements
GPSdt = 1/10 #time step between GPS updates
ACCdt = 1/50 #time step between Accelerometer updates

init_pos = 0.0
init_vel = 10.0
init_acc = 0.0

px = 0.0 # x Position Start
py = init_pos # y Position Start
pz = 0.0 # z Position Start

vx = 0.0 # m/s Velocity at the beginning
vy = init_vel # m/s Velocity
vz = 0.0 # m/s Velocity

Xr = []
Xa = []
Yr = []
Ya = []
Zr = []
Za = []
GPS = []
ACC = []
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
    
    if i % 2 == 0: #If the time has an accelerometer update
        ACC.append(1)
    else:
        ACC.append(0)
        
    if i % 10 == 0: #If the time has a GPS update
        GPS.append(1)
    else:
        GPS.append(0)

good = np.vstack((Xr,Yr,Zr))    
    
#Add Noise to the position measurement
spr= 1 # Sigma for position noise

Xmr = Xr + spr * (np.random.randn(m))
Ymr = Yr + spr * (np.random.randn(m))
Zmr = Zr + spr * (np.random.randn(m))

#Add Noise to the acceleration measurement
spa= .1 # Sigma for position noise

Xma = Xa + spa * (np.random.randn(m))
Yma = Ya + spa * (np.random.randn(m))
Zma = Za + spa * (np.random.randn(m))

measurements = np.vstack((Xmr,Ymr,Zmr,Xma,Yma,Zma))

#Make the initial state
x = np.matrix([0.0, init_pos, 0.0, 0.0, init_vel, 0.0, 0.0, init_acc, 0.0]).T
              
#Preallocation for plotting
xt = []
yt = []
zt = []

#Run the Kalman Filter
for filterstep in range(m):    
    # Time Update (Prediction)
    # ========================
    # Project the state ahead
    x = A*x + B*u
    
    # Project the error covariance ahead
    P = A*P*A.T + Q    
    
    #Make the Measurement Matrix
    if GPS[filterstep] == 1 and ACC[filterstep] == 1:
        H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    if GPS[filterstep] == 1 and ACC[filterstep] == 0:
        H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    if GPS[filterstep] == 0 and ACC[filterstep] == 1:
        H = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    if GPS[filterstep] == 0 and ACC[filterstep] == 0:
        H = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        
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
    
#Show overlayed position
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111, projection='3d')
ax.plot(xt,yt,zt, label='GPS and Accelerometer Kalman Filter Estimate')
#ax.plot(Xmr,Ymr,Zmr, label='GPS Measurement')
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
dist_measurement = np.sqrt((good[0,:]-Xmr)**2 + (good[1,:]-Ymr)**2 + (good[2,:]-Zmr)**2)
dist_both = np.sqrt((good[0,:]-xt)**2 + (good[1,:]-yt)**2 + (good[2,:]-zt)**2)
print('Estimated Position of GPS Measurement is %.2fm away from ball position.' % dist_measurement[-1])
print('Estimated Position of GPS and Accelerometer Kalman Filter is %.2fm away from ball position.' % dist_both[-1])