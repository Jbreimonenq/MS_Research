# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

u = sio.loadmat('Accel_Data')
z = sio.loadmat('Vel_Data')
u = np.array(u['a'])
z = np.array(z['v'])

def kf_predict(F,G,V,u,mu_1,sigma_1,Q_val = 0.1):
    Q = Q_val * np.eye(len(V))
    mu_bar = F*mu_1 + G*u
    sigma_bar = F*sigma_1*np.transpose(F) + V*Q*np.transpose(V)
    return mu_bar, sigma_bar

def kf_update(sigma_bar,mu_bar,z,C,R_val=0.1):
    R = R_val*np.eye(len(C))
    K = sigma_bar*C*(C*sigma_bar*np.transpose(C)+R)**-1
    mu = mu_bar + K*(z-C*mu_bar)
    sigma = sigma_bar-K*C*sigma_bar
    return mu, sigma

m = 1000
b = 50
time = np.zeros(len(u)-1)
F = np.array([1+.1*(-50/1000)])
G = np.array([.1*(1/1000)])
C = np.array([1])
V = np.array([.8])
mu_1 = 0
sigma_1 = 0
mu_test = np.zeros(len(u)-1)
mu = np.array([[]])
sigma = np.zeros(len(u)-1)
vel = np.zeros(len(u)-1)
t = 0
for i in range(len(u)-1):
    t = i * 0.1
    #
    mu_bar, sigma_bar = kf_predict(F, G, V, u[i], mu_1, sigma_1,0.5)
    mu_1, sigma_1 = kf_update(sigma_bar, mu_bar, z[i], C)
    print('mu = ', mu_1)
    mu_test[i] = mu_1 
    sigma[i] = sigma_1
    time[i] = t
    vel[i] = z[i] 
    
print('time = ',np.shape(time))
print('mu = ',np.shape(mu_test))
plt.plot(time,mu_test,linewidth=2.0)
plt.plot(time,vel,linewidth=2.0)
plt.ylabel('some numbers')
plt.show()

plt.plot(time,sigma,linewidth=2.0)
plt.ylabel('some numbers')
plt.show()

