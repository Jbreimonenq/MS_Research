import numpy as np
import matplotlib.pyplot as plt
#Classes and Functions --------------------------------------------------------
class REG_IEKS:
    def __init__(self,u_list, y_list, env, x0_hat, P, Q, R):
       
        self.n = env.dim_x
        self.m = R.shape[0]
        self.Q = Q
        self.R = R
        self.P = P
        self.x0_hat = x0_hat
        self.env = env
        self.y_list = y_list
        self.u_list = u_list
        
    def predict(self, u, x_c, x_f, P_f):
        F = self.env.Jacobian(x_c)
        #print(x_c.shape)
        #print(x_f.shape)
        #print(self.env.nextstate(x_c).shape)
        #u = self.env.control(x_c)
        #print(u)
        x_p = self.env.nextstate(x_c,u) + F @ (x_f - x_c)#print(self.x.shape[1])
        P_p = np.dot(np.dot(F, P_f), F.T) + self.Q #Predicted estimate covariance
        
        return x_p, P_p, F
   
    def update(self, z, x_c, x_p, P_p):
        H = self.env.JacobianMeasure(x_c)
        mu = self.env.measurement(x_c) + H @ (x_p - x_c)
        S = np.dot(H, np.dot(P_p, H.T)) + self.R #Innovation/ pre-fit residual covariance
        K = np.dot(np.dot(P_p, H.T), np.linalg.inv(S)) #Optimal Kalman gain
        x_f = x_p.reshape(self.n, 1) + np.dot(K, (z - mu)) #Updated state estimate
        P_f = P_p - np.dot(K,np.dot(S,K.T))
        
        return x_f, P_f    
    
    
    def smoothing(self, P_update, P_predict, x_update, x_predict, x_smooth, P_smooth, F):
        
        G = P_update @ F.T @ np.linalg.inv(P_predict)
        #print(G)
        x_sp = x_smooth - x_predict
        x_smoothing = x_update + np.dot(G, x_sp)
        P_sp = (P_smooth - P_predict)
        P_smoothing = P_update + np.dot(G,np.dot(P_sp,G.T))
        
        return x_smoothing, P_smoothing
    
    
    def run_R_EIKS(self, x_current):
        x_predict = []
        x_update = []
        P_predict = []
        P_update = []
        x_smoothing = []
        F = []
        T = len(self.y_list)
        for i in range(T):
            if i == 0:
                x = self.x0_hat
                P = self.P
            else:
                
                x, P, Fk = self.predict(self.u_list[i-1], x_current[i-1], x_update[i-1], P_update[i-1])
                
                F.append(Fk)
            #x = self.env.swing(x)
           # x[1] = self.env.clamp(x[1],-6,6)
            x_predict.append(x.reshape((self.n,1)))
            P_predict.append(P)
            measure = self.y_list[i]
            #print('x_current = ',x_current[i])
            x, P = self.update(measure, x_current[i], x_predict[i], P_predict[i])
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



