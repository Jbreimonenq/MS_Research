# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 23:23:04 2022

@author: reimoj
"""
import numpy as np
import matplotlib.pyplot as plt 
from Kalman_Smoother import REG_IEKS
from lorentz import lorentz
from EKF_Plotter import subplotIEKS as plot
#from EKF_Plotter import plotIEKS as plot

#Start of Main Code -----------------------------------------------------------
#Defining Variables
dt = 0.01
sim_t = 3
sigma = 13
row = 10
beta = 8/3

H = np.array([[1, 0, 0]])
in_states = len(H)
Q = 1e-6 * np.eye(3) 
R = 1e-0* np.eye(in_states)
P = 1e0*np.eye(3)
mean = np.zeros(in_states)
x0 = np.array([1,1,1]).reshape((3, 1))

states = 3


#Run Code
#x = env.nextstate(x0)
#j = env.Jacobian(x0)
#print(j)
env = lorentz(sigma, row, beta, H, dt)



# Create measurement
T = int(np.round((sim_t+dt)/dt))
s = x0.copy()
y_list = []
GT_state = []
for i in range(T):
    y = env.measurement(s) + np.random.multivariate_normal(mean, R)
    y_list.append(y)
    GT_state.append(s)
    s = env.nextstate(s)

kf = REG_IEKS(y_list, env, x0, P, Q, R)
x_current = []

for i in range(T):
    x = x0.copy()
    x[0, 0] = y_list[i][0, 0].copy()
    x_current.append(x)

#x_current = GT_state
xs = kf.run_R_EIKS()#x_current
#xs = x_current

#xp = np.stack(xp, axis=0)
#xs = np.stack(xs, axis=0)
#a = np.stack(a, axis=0)
#print(xs)



#print('Cost After = ', cost[1][:,0])

fig, axs = plt.subplots(3)
print('Plotting...')
plot(axs[0], dt = dt, m_value = y_list, p_value = xs, a_value = GT_state, state = 0)
plot(axs[1], dt = dt, p_value = xs, a_value = GT_state, state = 1) 
plot(axs[2], dt = dt, p_value = xs, a_value = GT_state, state = 2)
axs[0].title.set_text('State 1 of Lorentz System')
axs[1].title.set_text('State 2 of Lorentz System')
axs[2].title.set_text('State 3 of Lorentz System')



plt.figure(2)
x = np.arange(1,len(cost[1][:,0])+1)
y = cost[1][:,0]
plt.plot(x, y, marker='o')
plt.title('Cost vs Iterations')
plt.grid()
plt.text(x[0], y[0], y[0][0], size=12)
plt.text(x[-1], y[-1], y[-1][0], size=12)
print('Done.')
plt.show()

#print('Cost After = ', cost[1][:,0])
