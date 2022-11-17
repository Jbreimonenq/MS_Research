import numpy as np
import matplotlib.pyplot as plt
#Classes and Functions --------------------------------------------------------
class GN_IEKS_ILS:
    def __init__(self, y_list, env, x0, P, Q, R):
       
        self.n = env.dim_x
        self.m = R.shape[0]
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x0
        self.env = env
        self.y_list = y_list
        
        
    def predict(self, u = 0):
        
        F = self.env.Jacobian(self.x)#print(self.x.shape[1])
        self.x = self.env.nextstate(self.x)# + np.dot(self.B, u)  #Predicted state estimate
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q #Predicted estimate covariance
        
        return self.x, self.P, F
   
    def update(self,z):
        H = self.env.JacobianMeasure(z)
        y = z.reshape(self.m,1) - self.env.measurement(self.x).reshape((self.m, 1)) #Innovation/ pre-fit residual
        S = np.dot(H, np.dot(self.P, H.T)) + self.R #Innovation/ pre-fit residual covariance
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S)) #Optimal Kalman gain
        self.x = self.x.reshape(self.n, 1) + np.dot(K, y) #Updated state estimate
        IKH = (np.eye(self.n) - np.dot(K, H))
        self.P = self.P - np.dot(K,np.dot(S,K.T))#np.dot(IKH,self.P) #np.dot(IKH,np.dot(self.P,IKH.T)) + np.dot(K,np.dot(R,K.T)) # Updated estimate covariance
        
        return self.x, self.P
        
    def smoothing(self, P_update, P_predict, x_update, x_predict, x_smooth, P_smooth, F):
        
        G = P_update @ F.T @ np.linalg.inv(P_predict)
        #print(G)
        x_sp = x_smooth - x_predict
        x_smoothing = x_update + np.dot(G, x_sp)
        P_sp = (P_smooth - P_predict)
        P_smoothing = P_update + np.dot(G,np.dot(P_sp,G.T))
        
        return x_smoothing, P_smoothing
    
    
    def run_R_EIKS(self):
        x_predict = []
        x_update = []
        P_predict = []
        P_update = []
        x_smoothing = []
        F = []
        T = len(self.y_list)
        for i in range(T):
            measure = self.y_list[i]
            x, P, Fk = self.predict()
            x_predict.append(x.reshape((3,1)))
            F.append(Fk)
            P_predict.append(P)
            x, P = self.update(measure)
            x_update.append(x)
            P_update.append(P)
            

        P_smooth = P_update[-1]
        x_smooth = x_update[-1]
        #print(x_predict[-1])
        x_smoothing.append(x_smooth)

        for i in range(len(x_predict)-1):
            #print(x_update[-1-i])
            x_smooth, P_smooth = self.smoothing(P_update[-2-i], P_predict[-1-i], x_update[-2-i], x_predict[-1-i], x_smooth, P_smooth, F[-1-i])
            x_smoothing.append(x_smooth)
        
        x_smoothing.reverse()
        return  x_predict, x_smoothing



'''
#Inexact Line Search
    def dirDerivative(self):
        
        for i in range():
            sum1 = + sum1 
              
        
        d = 1
    
    def ILS(self, kf, mean, sim_t, dt, s, states):
        alpha = 1
        del_x = 0
        xp, xs, m, a = kf.run_R_EIKS(kf, mean, sim_t, dt, s, states)
        del_x = xs - del_x
        d = dirDerivative()
        
        while  >= + np.dot(alpha,np.dot(c1,d)
        alpha = np.dot(tau,alpha)
        '''
        


