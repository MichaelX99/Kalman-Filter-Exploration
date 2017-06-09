#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 23:29:07 2016

@author: mike
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

b = 10

both = np.zeros((b,1))
gps = np.zeros((b,1))
measure = np.zeros((b,1))

for it in range (b):
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
    T = 100 # s measuremnt time
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
    GPS = []
    ACC = []
    for i in range(int(m)):
        accx = 0
    
        vx += accx*dt
        px += vx*dt
        
        accz = 0
        vz += accz*dt
        pz += vz*dt
        
        accy = init_acc
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
    
        
    #Say how far away we actually are
    dist_measurement = np.sqrt((good[0,:]-Xmr)**2 + (good[1,:]-Ymr)**2 + (good[2,:]-Zmr)**2)
    dist_both = np.sqrt((good[0,:]-xt_both)**2 + (good[1,:]-yt_both)**2 + (good[2,:]-zt_both)**2)
    dist_gps = np.sqrt((good[0,:]-xt_gps)**2 + (good[1,:]-yt_gps)**2 + (good[2,:]-zt_gps)**2)
        
    both[it] = dist_both[-1]
    gps[it] = dist_gps[-1]
    measure[it] = dist_measurement[-1]    
    print(it)

#Show overlayed position
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
    
both_avg = np.average(both)
gps_avg = np.average(gps)
measure_avg = np.average(measure)
print('Average Estimated Position of GPS Measurement is %.2fm.' % measure_avg)
print('Average Estimated Position of GPS and Accelerometer Kalman Filter is %.2fm.' % both_avg)
print('Average Estimated Position of GPS Kalman Filter is %.2fm.' % gps_avg)

both_med = np.median(both)
gps_med = np.median(gps)
measure_med = np.median(measure)
print('Median Estimated Position of GPS Measurement is %.2fm.' % measure_med)
print('Median Estimated Position of GPS and Accelerometer Kalman Filter is %.2fm.' % both_med)
print('Median Estimated Position of GPS Kalman Filter is %.2fm.' % gps_med)