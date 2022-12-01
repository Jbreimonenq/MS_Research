# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:39:19 2022

@author: reimoj
"""

import numpy as np
import matplotlib.pyplot as plt

class linSys:
    def __init__ (self, H, dt):
        self.dt = dt
        self.dim_x = 2
        self.H = H
    
    def nextstate(self, x):
        x = x.reshape(self.dim_x)
        next_x = np.zeros((self.dim_x, 1))
        #print(x.shape)
        next_x[0] = x[0] + self.dt*x[1]
        next_x[1] = x[1]
        #print('x = ',x)
        return next_x
    
        
    def Jacobian(self, x):
        x = x.reshape(self.dim_x)
        x0 = x[0]
        x1 = x[1]
        f = np.array([[1, self.dt],
                      [0, 1]])
        
        return f

    def measurement(self, x):
        z = self.H @ x
        return z
    
    def JacobianMeasure(self, x):
        return self.H