import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


class KF:
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):
       
        if(F is None or H is None):
            raise ValueError("No F or H")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0
       
        mean = np.array([0, 0])
        cov = np.array([[1, 0], [0, 1]])
        w = np.random.multivariate_normal(mean, cov)
       
    def predict(self, u = 0):
        #print(self.x.shape)
        self.x = np.dot(self.F, self.x)# + np.dot(self.B, u)  #Predicted state estimate
        #print(self.x.shape)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q #Predicted estimate covariance
        #print(self.P)
        return self.x
   
    def update(self,z):
        #print(np.dot(self.H,self.x).shape)
        y = z.reshape(2,1) - np.dot(self.H, self.x) #Innovation/ pre-fit residual
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R #Innovation/ pre-fit residual covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) #Optimal Kalman gain
        self.x = self.x + np.dot(K, y) #Updated state estimate
        IKH = (np.eye(self.n) - np.dot(K, self.H))
        self.P = np.dot(IKH,self.P)#np.dot(IKH,np.dot(self.P,IKH.T))+ np.dot(K,np.dot(R,K.T)) # Updated estimate covariance
        #print(np.dot(K,y))
        #print(y.shape)
       
       
   

dt = 0.01
sim_t = 10
theta0 = 5*(np.pi/180)
A = np.array([[0, 1],[-9.81*theta0, 0]])
F = np.eye(len(A))+ np.dot(dt, A)
B = np.array([.5*dt**2, dt]).reshape(2, 1)
H = np.array([[1,0],[0,1]])
Q = np.dot(np.array([[(dt**4)/4, (dt**3)/2], [(dt**3)/2, (dt**2)]]), 0.1**2)
R = np.eye(2)
P = 1e-6*np.array([[1,0],[0,1]])
x0 = np.array([[theta0],[0]])

mean = np.array([0, 0])
#cov = np.array([[1, 0], [0, 1]])
kf = KF(F = F, B = B, H = H, Q = Q, R = R, P=P, x0=x0)
predictions = np.zeros((int(np.round((sim_t+dt)/dt)),2))
measurements = np.zeros((int(np.round((sim_t+dt)/dt)),2))
mes_clean = []
s = np.array([theta0, 0])

for i in range(int(np.round((sim_t+dt)/dt))):
    
    y = H @ s + np.random.multivariate_normal(mean, R)
    prediction = kf.predict()
    measurements[i] = y
    meas_clean = H @ s
    mes_clean.append(meas_clean)
    kf.update(measurements[i])
    #print(prediction[0])
    predictions[i] = kf.x.reshape(2)
    s = F @ s
#print(predictions)
mes_clean = np.array(mes_clean)





'''Plots for each state of the Measurement, Prediction, and True value.'''
for i in range(len(F)):
    #plt.plot(np.dot(dt,range(len(measurements))), measurements[:,i], label = 'Measurements')
    plt.plot(np.dot(dt,range(len(predictions))), predictions[:,i], 'r--', label = 'Kalman Filter Prediction')
    plt.plot(np.dot(dt,range(len(mes_clean))), mes_clean[:,i], 'g', label = 'True Value')
    plt.title(f'State %d'% (i+1))
    plt.legend()
    plt.grid()
    plt.show()