# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 23:16:26 2022

@author: reimoj
"""
import numpy as np
import matplotlib.pyplot as plt 
from InexactLineSearch_IEKS import GN_IEKS_ILS
from Pendulumn import pendulumn
from EKF_Plotter import subplotIEKS as plot

#Start of Main Code -----------------------------------------------------------
#Defining Variables
m = 1
l = 1
dt = 0.01
sim_t = 10
mean = np.array([0])
theta0 = 0*(np.pi/180)
H = np.array([[1,0]])
Q = 1e-2 * np.eye(2) 
R = np.eye(1)
P = 1e-12 * np.eye(2)
x0 = np.array([theta0,0]).reshape((2, 1))
states = 2

umax = 15
xdes = np.pi
#Run Code
env = pendulumn(m,l,H,umax,xdes,dt)



# Create measurement
T = int(np.round((sim_t+dt)/dt))
s = x0.copy()
y_list = []
GT_state = []
for i in range(T):
    u = env.control(s)
    y = env.measurement(s) + np.random.multivariate_normal(mean, R)
    y_list.append(y)
    GT_state.append(s)
    s = env.nextstate(s,u)

kf = GN_IEKS_ILS(y_list, env, x0, P, Q, R)
x_current = []

for i in range(T):
    x = x0.copy()
    x[0] = y_list[i].copy()
    x_current.append(x)

#x_current = GT_state
xs, cost = kf.solve(x_current)
#xs = x_current

#xp = np.stack(xp, axis=0)
#xs = np.stack(xs, axis=0)
#a = np.stack(a, axis=0)
#print(xs)



#print('Cost After = ', cost[1][:,0])

fig, axs = plt.subplots(2)
print('Plotting...')
plot(axs[0], dt = dt, m_value = y_list, p_value = xs, a_value = GT_state, state = 0)
plot(axs[1], dt = dt, p_value = xs, a_value = GT_state, state = 1) 
axs[0].title.set_text('Time Evolution of \u03F4')
#axs[0].axhline(y = np.pi, color = 'k', linestyle = ':')
axs[0].axes.xaxis.set_ticklabels([])
axs[1].title.set_text('Time Evolution of \u03C9')

print(repr(cost))

plt.figure(2)
x = range(len(cost))
y = cost
plt.plot(x, y, marker='o')
plt.title('Cost vs Iterations')
plt.grid()
for i, v in enumerate(cost):
    if i <= 1:
        plt.text(i, v+40000, "%d" %v, ha="left")
    elif i == len(cost)-1:
        plt.text(i, v+75000, "%d" %v, ha="center")

print('Done.')
plt.show()

#print('Cost After = ', cost[1][:,0])