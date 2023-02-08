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
     
    def control(self, x):
        omega_des = 0
        Kp = 10
        Ki = 0.9
        
        u = -Kp*(x[0,0]-self.theta_des) - Ki*(x[1,0]-omega_des)
        return u
    
    def nextstate(self, x, u=0):
        x_next = x[0,0] + self.dt*(x[1,0])
        v_next = (x[1,0] + self.dt*(u-self.g*np.sin(x[0,0]))) + u
        X_next = np.array([[x_next],[v_next]])
        return X_next
        
    def Jacobian(self, x):
        
        f = np.zeros((2,2))
        f[0,0] = 1
        f[0,1] = self.dt
        f[1,0] = -self.g*self.dt*np.cos(x[0])
        f[1,1] = 1
        return f
        
    def measurement(self, x):
        z = self.H @ x
        return z
    
    def JacobianMeasure(self, x):
        H_Jacobian = np.array([np.cos(x[0,0]),0]).reshape(1, 2)
        return self.H

