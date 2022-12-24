# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 10:23:02 2022

@author: reimoj
"""
import numpy as np
import matplotlib.pyplot as plt

class pendulumn:
    def __init__ (self, m, l, H, umax, theta_des, dt):
        self.dt = dt
        self.g = 9.81
        self.l = l
        self.m = 1
        self.H = H
        self.umax = umax
        self.theta_des = theta_des
        self.dim_x = 2
    
    def clamp(self, n, minn, maxn):
        if n > maxn:
            n = maxn
        elif n < minn:
            n = minn
        else:
            n = n
        return n
    def swing(self, x):
        if 0 <= x[0] <= 2*np.pi:
            x[0] = x[0]
        elif 0 > x[0]:
            x[0] = x[0] + (2*np.pi)
        else:
            x[0] = x[0] - (2*np.pi) 
        return x
    def control(self, x):
        omega_des = 0
        Kp = 12
        Ki = 1.5
        #x = self.swing(x)
        u = -Kp*(x[0]-self.theta_des) - Ki*(x[1]-omega_des)
        u = self.clamp(u, -self.umax, self.umax)
        return u
    
    def nextstate(self, x, u=0):
        
        x = x + self.dt*np.array([x[1], u-(self.g/self.l)*np.sin(x[0])])
        x = self.swing(x)
        
        
        return x
        
    def Jacobian(self, x):
        f = np.zeros((2,2))
        f[0,0] = 1
        f[0,1] = self.dt
        f[1,0] = -(self.g/self.l)*self.dt*np.cos(x[0])
        f[1,1] = 1
        
        return f
        
    def measurement(self, x):
        x = self.swing(x)
        z = self.H @ x
        return z
    
    def JacobianMeasure(self, x):
        return self.H


