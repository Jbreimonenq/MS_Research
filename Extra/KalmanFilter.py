# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import scipy.io as sio


def kf_predict(F,G,V,u,mu_1,sigma_1,Q_val):
    Q = Q_val
    mu_bar = F@mu_1 + G@u
    sigma_bar = F@sigma_1@np.transpose(F) + V@Q@np.transpose(V)
    return mu_bar, sigma_bar

def kf_update(sigma_bar,mu_bar,z,C,R_val):
    R = R_val
    K = sigma_bar@C@np.linalg.inv(C@sigma_bar@np.transpose(C)+R)
    mu = mu_bar + K@(z-C@mu_bar)
    sigma = sigma_bar-K@C@sigma_bar
    return mu, sigma

mean = [0, 0]
cov = [[1, 0], [0, 1]]
x, Q = np.random.multivariate_normal(mean, cov)
x, R = np.random.multivariate_normal(mean, cov)
u = np.array([[1,1,1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,1]])
time = np.zeros(len(u)-1) 
delta_t = 0.001
A = np.array([[0.9, 0.1],[0, 0.9]])
F = np.eye(len(A))+ delta_t @ A
G = 0 #np.array([[1, 0],[0, 1]])
C = np.array([[1, 0],[0, 1]])
V = np.array([[1],[1]])
mu_1 = 0
sigma_1 = 0
mu_test = np.zeros(len(u)-1)
mu = np.array([[]])
sigma = np.zeros(len(u)-1)
vel = np.zeros(len(u)-1)
xn1 = ([[0],[0]])




for i in range(len(u)-1):
    y = #measurement
    mu_bar, sigma_bar = kf_predict(F, G, V, u[:,i], mu_1, sigma_1, Q)
    mu_1, sigma_1 = kf_update(sigma_bar, mu_bar, y, C, R)
    print('mu = ', mu_1)
    mu_test[i] = mu_1 
    sigma[i] = sigma_1
    step[i] = i
     
    xn1 = F @ xn1 # next state
    
print('time = ',np.shape(time))
print('mu = ',np.shape(mu_test))
plt.plot(time,mu_test,linewidth=2.0)
plt.plot(time,vel,linewidth=2.0)
plt.ylabel('some numbers')
plt.show()

plt.plot(time,sigma,linewidth=2.0)
plt.ylabel('some numbers')
plt.show()

