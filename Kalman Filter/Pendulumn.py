# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 10:23:02 2022

@author: reimoj
"""
import numpy as np

class pendulumn:
    def __init__ (self, m, l, dt):
        self.dt = dt
        self.g = 9.81
        self.l = l
        self.m = 1
        self.dim_x = 2
    
    def nextstate(self, x):
        x = x + self.dt*np.array([x[1], -(self.g/self.l)*np.sin(x[0])])
        return x
        
    def Jacobian(self, x):
        f = np.zeros((2,2))
        f[0,0] = 1
        f[0,1] = self.dt
        f[1,0] = -(self.g/self.l)*self.dt*np.cos(x[0])
        f[1,1] = 1
        
        return f
