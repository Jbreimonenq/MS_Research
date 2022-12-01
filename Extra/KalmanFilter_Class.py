# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

class KF:
    def __init__(self, A, B, C, D, measurement, control, Q_val = 0.1, R_val=0.1):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.measurement = measurement
        self.control = control
        self.Q_val = Q_val
        self.R_val = R_val
        
    def predict(self,mu_1,sigma_1, con):
        Q = self.Q_val * np.eye(len(self.D))
        mu_bar = self.A*mu_1 + self.B*con
        sigma_bar = self.A*sigma_1*np.transpose(self.A) + self.D*Q*np.transpose(self.D)
        return mu_bar, sigma_bar
    
    def update(self,sigma_bar,mu_bar, meas):
        R = self.R_val*np.eye(len(self.C))
        K = sigma_bar*self.C*(self.C*sigma_bar*np.transpose(self.C)+R)**-1
        mu = mu_bar + K*(meas-self.C*mu_bar)
        sigma = sigma_bar-K*self.C*sigma_bar
        return mu, sigma
    
    def calculate(self):
        mu_1 = 0
        sigma_1 = 0
        mu = np.zeros(len(self.measurement)-1)
        #print(mu)
        sigma = np.zeros(len(self.control)-1)
        t = np.zeros(len(self.measurement)-1)
        measure = np.zeros(len(self.measurement)-1)
        cntr = np.zeros(len(self.control)-1)
        for i in range(len(self.measurement)-1):
            t[i] = i
            mu_bar, sigma_bar = self.predict(mu_1, sigma_1, self.control[i])
            mu_1, sigma_1 = self.update(mu_bar, sigma_bar, self.measurement[i])
            #print('mu = ',mu_1)
            mu[i] = mu_1 
            sigma[i] = sigma_1
            measure[i] = self.measurement[i]
            cntr[i] = self.control[i]
            
        plt.plot(t, mu, linewidth=2.0)
        plt.plot(t, measure,'--', linewidth=2.0)
        plt.ylabel('mu')
        plt.xlabel('Step')
        plt.legend(['Calculated', 'Measured'], loc='lower right')
        plt.show()
        
        plt.plot(t, sigma, linewidth=2.0)
        plt.ylabel('Sigma')
        plt.xlabel('Step')
        plt.legend(['Calculated'], loc='lower right')
        plt.show()
        return mu, sigma
 
u = sio.loadmat('Accel_Data')
z = sio.loadmat('Vel_Data')
u = np.array(u['a'])
z = np.array(z['v'])

F = np.array([1+.1*(-50/1000)])
G = np.array([.1*(1/1000)])
C = np.array([1])
V = np.array([.8])

CC_KF = KF(F,G,C,V,z,u)

CC_KF.calculate()
