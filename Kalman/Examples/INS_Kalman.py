#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 09:56:58 2016

@author: mike
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import math

def update(xr,xv,xa,yr,yv,ya,zr,zv,za):
    vx = xv + xa * dt
    vy = yv + ya * dt
    vz = zv + za * dt
   
    px = xr + vx * dt
    py = yr + vy * dt
    pz = zr + vz * dt
    
    new = [px,vx,xa,py,vy,ya,pz,vz,za]
    return new

#Init matrices
dt = .01
A = np.eye(9)
P = np.eye(9)
H = np.zeros((6,9))
R = np.eye(6)
I = np.eye(9)
G = np.zeros((9,1))
A[0,1] = dt
A[0,2] = pow(dt,2)/2
A[1,2] = dt
A[3,4] = dt
A[3,5] = pow(dt,2)/2
A[4,5] = dt
A[6,7] = dt
A[6,8] = pow(dt,2)/2
A[7,8] = dt
P *= 100
H[0,0] = 1
H[1,2] = 1
H[2,3] = 1
H[3,5] = 1
H[4,6] = 1
H[5,8] = 1
G[0] = pow(dt,2)/2
G[1] = dt
G[2] = 1
G[3] = pow(dt,2)/2
G[4] = dt
G[5] = 1
G[6] = pow(dt,2)/2
G[7] = dt
G[8] = 1

sa = 0.1
Q = np.dot(G,G.T)
Q *= pow(sa,2)

#Synthetically Create Position Measurement for Ball
Hz = 50.0 # Frequency of Vision System
dt = 1.0/Hz
T = 12.0 # s measuremnt time
m = int(T/dt) # number of measurements

px0 = 0.0 # x Position Start
py0 = 0.0 # y Position Start
pz0 = 0.0 # z Position Start

px1 = 0
py1 = 100
pz1 = 0

px2 = 5
py2 = 115
pz2 = 0

px3 = px2
py3 = 180
py3 = 0

vx = 0
vy = 0
vz = 0

vmax = 20
amax = 3

regime = 0

Xr = [0]
Xv = [0]
Xa = [0]
Yr = [0]
Yv = [0]
Ya = [0]
Zr = [0]
Zv = [0]
Za = [0]

for i in range(1,int(m)):
    if i < 5:
        Ya.append(50)
    else:
        Ya.append(0)
    Yv.append(Ya[i]*dt+Yv[i-1])
    Yr.append(Yv[i]*dt+Yr[i-1])    
    Xr.append(0)
    Xv.append(0)
    Xa.append(0)
    Zr.append(0)
    Zv.append(0)
    Za.append(0)

#result = [Xr[0], Xv[0], Xa[0], Yr[0], Yv[0], Ya[0], Zr[0], Zv[0], Za[0]]
#for i in range(1,int(m)):
#    #first portion
#    if regime == 0:
#        ax = 0
#        az = 0
#        if vy < vmax:
#            ay = amax
#        if vy > vmax:
#            ay = 0
#        
#    if regime == 1:
#        if Xr[i-1] < px2/2:
#            ax = amax
#        if Xr[i-1] >= px2/2 and Xr[i-1] < px2:
#            ax = -amax
#        if  Xr[i-1] >= px2:
#            ax = 0
#        ay = 0
#        az = 0
#        
#    if regime == 2:
#        ax = 0
#        ay = 0
#        az = 0
#        Xv[i-1] = 0
#
#    result = update(Xr[i-1],Xv[i-1],ax,Yr[i-1],Yv[i-1],ay,Zr[i-1],Zv[i-1],az)        
#    Xr.append(result[0])
#    Xv.append(result[1])
#    Xa.append(result[2])
#    Yr.append(result[3])
#    Yv.append(result[4])
#    Ya.append(result[5])
#    Zr.append(result[6])
#    Zv.append(result[7])
#    Za.append(result[8])
    
#    if Yr[i] < py1:
#        regime = 0
#    
#    if Xr[i] < px2 and Yr[i] >= py1:
#        regime = 1
#
#    if Xr[i] >= px3 and Xa[i] == 0:
#        regime = 2

fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111, projection='3d')        
ax.plot(Xr, Yr, Zr, c='gray')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Car Position (Actual)')

fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111, projection='3d')        
ax.plot(Xv, Yv, Zv, c='gray')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Car Velocity (Actual)')

fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111, projection='3d')        
ax.plot(Xa, Ya, Za, c='gray')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Car Acceleration (Actual)')

#Add Noise to the position measurement
spr = 1 # Sigma for position noise
Xmr = Xr #+ spr * (np.random.randn(m))
Ymr = Yr #+ spr * (np.random.randn(m))
Zmr = Zr

#fig = plt.figure(figsize=(16,9))
#ax = fig.add_subplot(111, projection='3d')        
#ax.scatter(Xmr, Ymr, Zmr, c='gray')
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#plt.title('Car Position (Sensor)')

#Add Noise to the position measurement
spa = .1 # Sigma for position noise
Xma = Xa #+ spa * (np.random.randn(m))
Yma = Ya #+ spa * (np.random.randn(m))
Zma = Za

#Make the measurement vector
measurements = np.vstack((Xmr,Xma,Ymr,Yma,Zmr,Zma))

#Make the initial state
x = np.matrix([Xmr[0], Xv[0], Xma[0], Ymr[0], Yv[0], Yma[0], Zmr[0], Zv[0], Zma[0]]).T

#Preallocation for plotting
xrf = []
xvf = []
xaf = []
yrf = []
yvf = []
yaf = []
zrf = []
zvf = []
zaf = []

#Run the Kalman Filter
for filterstep in range(int(m)):    
    # Time Update (Prediction)
    # ========================
    # Project the state ahead
    x = np.dot(A,x)
    
    # Project the error covariance ahead
    P = np.add(np.dot(A,np.dot(P,A.T)),Q)    
    
    # Measurement Update (Correction)
    # ===============================
    # Compute the Kalman Gain
    S = np.add(np.dot(H,np.dot(P,H.T)),R)
    K = np.dot(np.dot(P,H.T),np.linalg.pinv(S))
    
    # Update the estimate via z
    Z = measurements[:,filterstep].reshape(H.shape[0],1)
    y = np.subtract(Z,np.dot(H,x))                            # Innovation or Residual
    x = np.add(x,np.dot(K,y))
    
    # Update the error covariance
    P = np.dot(np.subtract(I,np.dot(K,H)),P)
    
    # Save states for Plotting
    xrf.append(float(x[0]))
    xvf.append(float(x[1]))
    xaf.append(float(x[2]))
    yrf.append(float(x[3]))
    yvf.append(float(x[4]))
    yaf.append(float(x[5]))
    zrf.append(float(x[6]))
    zvf.append(float(x[7]))
    zaf.append(float(x[8]))
    
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111, projection='3d')        
ax.plot(xrf, yrf, zrf, c='red')
ax.plot(Xr,Yr,Zr,c='blue')
ax.plot(Xmr,Ymr,Zmr,c='grey')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Car Position (Kalman Filter)')