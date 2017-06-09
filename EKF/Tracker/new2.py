#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 11:47:10 2017

@author: mike
"""

import numpy as np
import matplotlib.pyplot as plt
import math

#Synthetically Create Position Measurement for Ball
Hz = 100.0 # Frequency of Vision System
dt = 1.0/Hz
T = 100 # s measuremnt time
m = int(T/dt) # number of measurements

#make the error covariance matrix
P = np.eye(6)
P *= 100

#make the process noise covariance matrix
sACC = 8.8
sGPS = .5 * dt**2 * sACC
sCourse = .1 * dt
sVel = dt * sACC
Q = np.diag([sGPS**2,sGPS**2,sVel**2,sCourse**2,sACC**2,sACC**2])

#make the measurement jacobian
Jh = np.eye(6)

#make the measurement noise covariance
stdGPS = 5
stdSpeed = 3
stdYaw = .1
stdACC = 1
R = np.diag([stdGPS**2,stdGPS**2,stdSpeed**2,stdYaw**2,stdACC**2,stdACC**2])

#make identity matrix
I = np.eye(6)
    
init_pos = 0.0
init_vel = 10.0
init_acc = 10.0

px= 0.0 # x Position Start
py= init_pos # y Position Start

vx = 0.0 # m/s Velocity at the beginning
vy = init_vel # m/s Velocity

Xr = []
Yr = []
V = []
Psi = []
Xa = []
Ya = []
GPS = []
for i in range(int(m)):
    accx = 0
    
    vx += accx*dt
    px += vx*dt
        
    accy = init_acc
    vy += accy*dt
    py += vy*dt
        
    Xr.append(px)
    Yr.append(py)
    V.append(math.sqrt(vx**2+vy**2))
    Psi.append(0)
    Xa.append(accx)
    Ya.append(accy)
    if i % 10 == 0:
        GPS.append(1)
    else:
        GPS.append(0)

good = np.vstack((Xr,Yr))    
    
#Add Noise to the position measurement
spr= 1 # Sigma for position noise
Xmr = Xr + spr * (np.random.randn(m))
Ymr = Yr + spr * (np.random.randn(m))

#Add Noise to velocity measurement
spv = .1
Vm = V + spv * np.random.randn(m)

#Add Noise to course measurement
spc = .1
Psim = Psi + spc * np.random.randn(m)

#Add Noise to the acceleration measurement
spa= .01 # Sigma for position noise
Xma = Xa + spa * (np.random.randn(m))
Yma = Ya + spa * (np.random.randn(m))

measurements_both = np.vstack((Xmr,Ymr,Vm,Psim,Xma,Yma))

# Preallocation for Plotting
xt = []
yt = []
vt = []
psit = []
xddt = []
yddt = []

#make the initial state
x = np.matrix([0.0, init_pos, init_vel, 0.0, 0.0, init_acc]).T

for filtersteps in range(int(m)):
    # Time Update (Prediction)
    # ========================
    # Project the state ahead
    x[0] = x[0]+.5*dt**2*x[4]+dt*x[2]*np.sin(x[3])
    x[1] = x[1]+.5*dt**2*x[5]+dt*x[2]*np.cos(x[3])
    x[2] = x[2]
    x[3] = x[3]
    x[4] = x[4]
    x[5] = x[5]
    
    #Calculate the Jacobian
    a02 = dt*np.sin(x[3])
    a03 = dt*x[2]*np.cos(x[3])
    a04 = .5*dt**2
    a12 = dt*np.cos(x[3])
    a13 = -1*dt*x[2]*np.sin(x[3])
    a15 = .5*dt**2
    a24 = dt*x[4]*np.power(x[4]**2+x[5]**2,-1/2)
    a25 = dt*x[5]*np.power(x[4]**2+x[5]**2,-1/2)
    Ja = np.matrix([[1.0, 0.0, a02, a03, a04, 0.0],
                    [0.0, 1.0, a12, a13, 0.0, a15],
                    [0.0, 0.0, 1.0, 0.0, a24, a25],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    
    #Project the error covariance ahead
    P = Ja*P*Ja.T + Q
    
     # Measurement Update (Correction)
    # ===============================
    # Measurement Function
    if GPS[filtersteps] == 1:
        hx = np.matrix([[float(x[0])],
                        [float(x[1])],
                        [float(x[2])],
                        [float(x[3])],
                        [float(x[4])],
                        [float(x[5])]])

        Jh = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    else:
        hx = np.matrix([[float(0)],
                        [float(0)],
                        [float(x[2])],
                        [float(x[3])],
                        [float(x[4])],
                        [float(x[5])]])

        Jh = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    
    S = Jh*P*Jh.T + R
    K = (P*Jh.T) * np.linalg.pinv(S)

    # Update the estimate via
    Z = measurements_both[:,filtersteps].reshape(Jh.shape[0],1)
    y = Z - hx                         # Innovation or Residual
    x = x + (K*y)

    # Update the error covariance
    P = (I - (K*Jh))*P
    
    xt.append(float(x[0]))
    yt.append(float(x[1]))
    vt.append(float(x[2]))
    psit.append(float(x[3]))
    xddt.append(float(x[4]))
    yddt.append(float(x[5]))

#Show overlayed position
fig = plt.figure(figsize=(12,9))
plt.subplot(221)
# EKF State
plt.plot(xt,yt, label='EKF Position', c='g', lw=1)
plt.plot(Xr,Yr, label='EKF Position', c='r', lw=1)
plt.plot(Xmr,Ymr, label='EKF Position', c='b', lw=.06)
plt.title('Position')

fig = plt.figure(figsize=(12,9))
plt.subplot(221)
plt.plot(psit, label='EKF Position', c='g', lw=5)
plt.plot(Psi, label='EKF Position', c='r', lw=5)
plt.title('Psi')