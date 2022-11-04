import numpy as np
import matplotlib.pyplot as plt
#Classes and Functions --------------------------------------------------------
class EKF:
    def __init__(self, B = None, H = None, Q = None, R = None, P = None, x0 = None, env = None):
       
        #if(F is None or H is None):
        #    raise ValueError("No F or H")

        self.n = env.dim_x
        self.m = H.shape[0]
        self.F = np.zeros((self.n,self.n))
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n,1)) if x0 is None else x0
        self.env = env
        #print(self.x)
        
        
    def predict(self, u = 0):
        #print(self.x.shape)
        
        self.F = self.env.Jacobian(self.x)#print(self.x.shape[1])
        self.x = self.env.nextstate(self.x)# + np.dot(self.B, u)  #Predicted state estimate
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q #Predicted estimate covariance
        #print(self.P)
        
        return self.x
   
    def update(self,z):
        #print(np.dot(self.H,self.x).shape)
        #print("Hx", np.dot(self.H, self.x).shape)
        y = z.reshape(self.m,1) - np.dot(self.H, self.x).reshape((self.m, 1)) #Innovation/ pre-fit residual
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R #Innovation/ pre-fit residual covariance
        #print(y.shape)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) #Optimal Kalman gain
        #print("K", K.shape)
        #print("y", y.shape)
        #print("x", self.x)
        self.x = self.x.reshape(self.n, 1) + np.dot(K, y) #Updated state estimate
        IKH = (np.eye(self.n) - np.dot(K, self.H))
        self.P = np.dot(IKH,self.P) #np.dot(IKH,np.dot(self.P,IKH.T)) + np.dot(K,np.dot(R,K.T)) # Updated estimate covariance
        #print(np.dot(K,y))
        #print(y.shape)

    def run_EKF(self, kf, mean, sim_t, dt, s, states):
        predictions = np.zeros((int(np.round((sim_t+dt)/dt)),states))
        measurements = np.zeros((int(np.round((sim_t+dt)/dt)),self.R.shape[0]))
        GT_state = []
        
        for i in range(int(np.round((sim_t+dt)/dt))):
            measurements[i] = self.H @ s + np.random.multivariate_normal(mean, self.R)
            prediction = kf.predict()
            GT_state.append(s)
            kf.update(measurements[i])
            #print(prediction[0])
            predictions[i] = prediction.reshape(states)
            s = self.env.nextstate(s)
            
        GT_state = np.array(GT_state)
        
        return  predictions, measurements, GT_state   
        
