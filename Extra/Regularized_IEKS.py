import numpy as np
import matplotlib.pyplot as plt
#Classes and Functions --------------------------------------------------------
class R_IEKS:
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
        
        return self.x, self.P, self.F
   
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
        self.P = self.P - np.dot(K,np.dot(S,K.T))#np.dot(IKH,self.P) #np.dot(IKH,np.dot(self.P,IKH.T)) + np.dot(K,np.dot(R,K.T)) # Updated estimate covariance
        #print(np.dot(K,y))
        #print(y.shape)
        
        return self.x, self.P
        
    def smoothing(self, P_update, P_predict, x_update, x_predict, x_smooth, P_smooth, F):
        
        G = P_update @ F.T @ np.linalg.inv(P_predict)
        print(G)
        x_sp = x_smooth - x_predict
        x_smoothing = x_update + np.dot(G, x_sp)
        P_sp = (P_smooth - P_predict)
        P_smoothing = P_update + np.dot(G,np.dot(P_sp,G.T))
        
        return x_smoothing, P_smoothing
    
    
    def run_R_EIKS(self, kf, mean, sim_t, dt, s, states):
        x_predict = []
        x_update = []
        P_predict = []
        P_update = []
        x_smoothing = []
        measurements = []
        GT_state = []
        F = []
        
        for i in range(int(np.round((sim_t+dt)/dt))):
            measure = self.H @ s + np.random.multivariate_normal(mean, self.R)
            measurements.append(measure)
            x, P, Fk = kf.predict()
            x_predict.append(x.reshape((3,1)))
            F.append(Fk)
            P_predict.append(P)
            GT_state.append(s)
            x, P = kf.update(measure)
            x_update.append(x)
            P_update.append(P)
            s = self.env.nextstate(s)
            
        GT_state = np.array(GT_state)
        P_smooth = P_update[-1]
        x_smooth = x_update[-1]
        #print(x_predict[-1])
        
        for i in range(len(x_predict)):
            #print(x_update[-1-i])
            x_smooth, P_smooth = kf.smoothing(P_update[-1-i], P_predict[-1-i], x_update[-1-i], x_predict[-1-i], x_smooth, P_smooth, F[-1-i])
            x_smoothing.append(x_smooth)
        
        x_smoothing.reverse()
        return  x_predict, x_smoothing, measurements, GT_state
    
    def dirDerivative(self):
        
        d = 


