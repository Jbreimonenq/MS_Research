import numpy as np
import matplotlib.pyplot as plt

class KUKA:
    def __init__ (self, m, l, H, des_position, dt):
        self.dt = dt
        self.g = 9.81
        self.l = l
        self.m = 1
        self.H = H
        self.dim_x = 7
        self.des_position = des_position
        
    
    def control(self, des_position):
        return u
    
    def nextstate(self, x, u=0):
        return X_next
        
    def Jacobian(self, x):
        return f
        
    def measurement(self, x):
        z = self.H @ x
        return z
    
    def JacobianMeasure(self, x):
        return self.H