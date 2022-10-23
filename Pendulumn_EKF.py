import numpy as np
import matplotlib.pyplot as plt
from Pendulumn import pendulumn
from EKF import EKF
#
def run_EKF(sim_t, dt, s):
    predictions = np.zeros((int(np.round((sim_t+dt)/dt)),2))
    measurements = np.zeros((int(np.round((sim_t+dt)/dt)),2))
    mes_clean = []
    
    for i in range(int(np.round((sim_t+dt)/dt))):
        y = H @ s + np.random.multivariate_normal(mean, R)
        prediction = kf.predict()
        measurements[i] = y
        meas_clean = H @ s
        mes_clean.append(meas_clean)
        kf.update(measurements[i])
        #print(prediction[0])
        predictions[i] = kf.x.reshape(2)
        s = env.nextstate(s)
        
    mes_clean = np.array(mes_clean)
    
    return  predictions, measurements, mes_clean
#Start of Main Code -----------------------------------------------------------
#Defining Variables
m = 1
l = 1
dt = 0.001
sim_t = 1
theta0 = 2.5
H = np.eye(2)#
Q = 0 * np.eye(2) 
R = np.eye(2)
P = 1e3*np.eye(2)
mean = np.array([0, 0])
x0 = np.array([[theta0],[0]])
s = np.array([theta0,0])

#Run Code
env = pendulumn(m,l,dt)
kf = EKF( H = H, P = P, Q=Q, env = env, x0 = x0)
predictions, measurements, mes_clean = run_EKF(sim_t, dt, s)



#Plots ------------------------------------------------------------------------

for i in range(2):
    plt.plot(np.dot(dt,range(len(measurements))), measurements[:,i], label = 'Measured Value')
    plt.plot(np.dot(dt,range(len(predictions))), predictions[:,i], 'r--', label = 'Predicted Value')
    plt.plot(np.dot(dt,range(len(mes_clean))), mes_clean[:,i], 'g', label = 'Actual Value')
    plt.title(f'State %d'% (i+1))
    plt.legend()
    plt.grid()
    plt.show()

