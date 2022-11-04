# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 23:16:26 2022

@author: reimoj
"""
import numpy as np
import matplotlib.pyplot as plt 
from EKF import EKF
from Pendulumn import pendulumn
from EKF_Plotter import plot
 
#Start of Main Code -----------------------------------------------------------
#Defining Variables
m = 1
l = 1
dt = 0.01
sim_t = 3
mean = np.array([0])
theta0 = 50*(np.pi/180)
H = np.array([[1,0]])
Q = 1e-6 * np.eye(2) 
R = np.eye(1)
P = 1e4 * np.eye(2)
x0 = np.array([[theta0],[0]])
s = np.array([theta0,0])
states = 2

#Run Code
env = pendulumn(m,l,dt)
kf = EKF( H = H, P = P, Q=Q, R=R, env = env, x0 = x0)
p, m, a = kf.run_EKF(kf, mean, sim_t, dt, s, states)


#Plots ------------------------------------------------------------------------

plot(dt = dt, m_value = m, p_value = p, a_value = a, state = 0)
plot(dt = dt, p_value = p, a_value = a, state = 1) #,m_value = m, )