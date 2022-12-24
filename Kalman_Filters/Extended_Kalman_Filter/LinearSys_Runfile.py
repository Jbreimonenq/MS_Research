# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:52:29 2022

@author: reimoj
"""
import numpy as np
import matplotlib.pyplot as plt 
from InexactLineSearch_IEKS import GN_IEKS_ILS
from LinearSys import linSys
from EKF_Plotter import plotIEKS as plot


#Start of Main Code -----------------------------------------------------------
#Defining Variables
dt = 0.01
sim_t = 3

H = np.array([[1, 0]])
in_states = len(H)
Q = 1e-6 * np.eye(2) 
R = 1e-0* np.eye(in_states)
P = 1e0*np.eye(2)
mean = np.zeros(in_states)
x0 = np.array([10,-1]).reshape((2, 1))

states = 2


#Run Code
#x = env.nextstate(x0)
#j = env.Jacobian(x0)
#print(j)
env = linSys(H, dt)



# Create measurement
T = int(np.round((sim_t+dt)/dt))
s = x0
y_list = []
GT_state = []
for i in range(T):
    y = env.measurement(s) + np.random.multivariate_normal(mean, R)
    y_list.append(y)
    GT_state.append(s)
    s = env.nextstate(s)


kf = GN_IEKS_ILS(y_list, env, x0, P, Q, R)
x_current = [x0] * T

xs = kf.solve(x_current)

#xp = np.stack(xp, axis=0)
#xs = np.stack(xs, axis=0)
#a = np.stack(a, axis=0)
#print(xs)

plot(dt = dt, m_value = y_list, p_value = xs, a_value = GT_state, state = 0)
#plot(dt = dt, p_value = p, a_value = a, state = 1) 
#plot(dt = dt, p_value = p, a_value = a, state = 2)