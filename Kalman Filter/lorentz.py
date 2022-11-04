# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 22:17:55 2022

@author: reimoj
"""
import numpy as np
import matplotlib.pyplot as plt
from EKF import EKF

class lorentz:
    def __init__ (self, sigma, row, beta, dt):
        self.dt = dt
        self.sigma = sigma
        self.row = row
        self.beta = beta
        self.dim_x = 3
    
    def nextstate(self, x):
        x = x.reshape(3)
        next_x = np.zeros(3)
        print(x.shape)
        next_x[0] = x[0] + self.dt*(self.sigma*(x[1]-x[0]))
        next_x[1] = x[1] + self.dt*(x[0]*(self.row - x[2])-x[1])
        next_x[2] = x[2] + self.dt*((x[0]*x[1]-self.beta*x[2]))
        print('x = ',x)
        return next_x
    
        
    def Jacobian(self, x):
        x = x.reshape(3)
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        f = np.array([[-self.sigma, self.sigma, 0],
                      [self.row-x2, -1, x0*(self.row-1)],
                      [x1, x0, -self.beta]])
        
        return np.eye(3) + self.dt * f
