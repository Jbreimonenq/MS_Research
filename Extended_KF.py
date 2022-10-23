import numpy as np
from numpy.linalg import inv
import sympy as sym
import matplotlib.pyplot as plt
from Pendulumn import pendulumn
#Classes and Functions --------------------------------------------------------
class EKF:
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None, env = None):
       
        #if(F is None or H is None):
        #    raise ValueError("No F or H")

        self.n = env.dim_x
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = x0
        self.env = env
        
        
    def predict(self, u = 0):
        #print(self.x.shape)
        self.F = self.env.Jacobian(self.x)
        self.x = self.env.nextstate(self.x)# + np.dot(self.B, u)  #Predicted state estimate
        #print(self.x.shape[1])
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q #Predicted estimate covariance
        #print(self.P)
        return self.x
   
    def update(self,z):
        #print(np.dot(self.H,self.x).shape)
        #print(len(self.x[1]))
        y = z.reshape(self.x.shape[0],self.x.shape[1]) - np.dot(self.H, self.x) #Innovation/ pre-fit residual
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R #Innovation/ pre-fit residual covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) #Optimal Kalman gain
        self.x = self.x + np.dot(K, y) #Updated state estimate
        IKH = (np.eye(self.n) - np.dot(K, self.H))
        self.P = np.dot(IKH,np.dot(self.P,IKH.T)) + np.dot(K,np.dot(R,K.T)) # Updated estimate covariance
        #print(np.dot(K,y))
        #print(y.shape)
       
#Start of Main Code -----------------------------------------------------------
m = 1
l = 1
dt = 0.001
sim_t = 1
theta = np.pi
H = np.eye(2)#np.array([[1,0],[0,1]])
R = np.eye(2)
P = 1e-6 * np.eye(2)
mean = np.array([0, 0])

env = pendulumn(m,l,dt)
kf = EKF( H = H, R = R, P = P, env = env)
predictions = np.zeros((int(np.round((sim_t+dt)/dt)),2))
measurements = np.zeros((int(np.round((sim_t+dt)/dt)),2))
mes_clean = []
s = np.array([2,0])




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


#Plots ------------------------------------------------------------------------

for i in range(2):
    plt.plot(np.dot(dt,range(len(measurements))), measurements[:,i], label = 'Measured Value')
    plt.plot(np.dot(dt,range(len(predictions))), predictions[:,i], 'r--', label = 'Predicted Value')
    plt.plot(np.dot(dt,range(len(mes_clean))), mes_clean[:,i], 'g', label = 'Actual Value')
    plt.title(f'State %d'% (i+1))
    plt.legend()
    plt.grid()
    plt.show()
	
