# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 00:16:15 2022

@author: reimoj
"""
import numpy as np
import matplotlib.pyplot as plt 
from EKF import EKF
from lorentz import lorentz
from EKF_Plotter import plot

#Start of Main Code -----------------------------------------------------------
#Defining Variables
dt = 0.01
sim_t = 3
sigma = 13
row = 10
beta = 8/3

H = np.array([[1, 0, 0], [0, 1, 0]])
Q = 1e-6 * np.eye(3) 
R = 1e-0* np.eye(2)
P = 1e-3*np.eye(3)
mean = np.zeros(2)
x0 = np.array([1,1,1]).reshape((3, 1))
s = np.array([1,1,1])
states = 3


#Run Code
env = lorentz(sigma, row, beta, dt)
#x = env.nextstate(x0)
#j = env.Jacobian(x0)
#print(j)

kf = EKF( H = H, P = P, Q = Q, R=R, env = env, x0 = x0)
p, m, a = kf.run_EKF(kf, mean, sim_t, dt, s, states)

plot(dt = dt, m_value = m, p_value = p, a_value = a, state = 0)
plot(dt = dt, m_value = m, p_value = p, a_value = a, state = 1)
plot(dt = dt, p_value = p, a_value = a, state = 2)