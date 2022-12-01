import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos
from scipy.integrate import odeint
import scipy.io as sio
'''E-Kalman Filter for Simple Pendulumn'''
'''Jacobian for pendulumn J = np.array([[0, 1],[-(g/l)*cos(x1), 0]])'''
g = 9.81
l = 1

def equations(y0, t):
	theta, x = y0
	f = [x, -(g/l) * sin(theta)]
	return f

time = np.arange(0, 10.0, 0.001)

# initial conditions
initial_angle = 70
theta0 = np.radians(initial_angle)
x0 = np.radians(0.0)

# find the solution to the nonlinear problem
theta = odeint(equations, [theta0, x0],  time)


def ekf_predict(F,V,fx,mu_1,sigma_1,Q):
    mu_bar = mu_1 + delta_t*fx
    sigma_bar = F @ sigma_1 @ np.transpose(F) + V @ Q @ np.transpose(V)
    print("sigma_bar  =  ",sigma_bar)
    return mu_bar, sigma_bar

def ekf_update(sigma_bar,mu_bar,z,C,R):
    K = sigma_bar @ C @ np.linalg.inv(C @ sigma_bar @ np.transpose(C) + R)
    mu = mu_bar + K @ (z-C @ mu_bar)
    sigma = sigma_bar-K  @ C @ sigma_bar
    return mu, sigma

#mean = [0, 0]
#cov = [[1, 0], [0, 1]]
#x, Q = np.random.multivariate_normal(mean, cov)
#x, R = np.random.multivariate_normal(mean, cov)

mu = np.array([[70],[0]])
fx = np.array([[mu[0]],[mu[1]]])

V = 0
C = np.array([[1, 0],[0, 0]])
delta_t = 0.001
mu_1 =  np.zeros(2)
sigma_1 = np.zeros(2)
mu_test = np.zeros(len(time)-1)
mu = np.array([[]])
sigma = np.zeros(len(time)-1)
vel = np.zeros(len(time)-1)
xn1 = ([[1],[1]])



for i in range(len(time)):
    A = np.array([[0, 1],[-(g/l)*cos(theta[i]), 0]]) #Jacobian
    F = np.eye(2) + delta_t * A
    y = np.array([[theta[i],0],[0,0]])#measurement
    mu_bar, sigma_bar = ekf_predict(F, V, fx, 0, mu_1, sigma_1)
    mu_1, sigma_1 = ekf_update(sigma_bar, mu_bar, y, C,0)
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


