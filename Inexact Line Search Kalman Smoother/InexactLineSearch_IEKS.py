import numpy as np
import matplotlib.pyplot as plt
#Classes and Functions --------------------------------------------------------
class GN_IEKS_ILS:
    def __init__(self, y_list, env, x0_hat, P, Q, R):
       
        self.n = env.dim_x
        self.m = R.shape[0]
        self.Q = Q
        self.R = R
        self.P = P
        self.x0_hat = x0_hat
        self.env = env
        self.y_list = y_list
        
        
    def predict(self, x_c, x_f, P_f):
        F = self.env.Jacobian(x_c)
        #print(x_c.shape)
        #print(x_f.shape)
        #print(self.env.nextstate(x_c).shape)
        x_p = self.env.nextstate(x_c) + F @ (x_f - x_c)#print(self.x.shape[1])
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
    
    
    def step(self, x_current):
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
                x, P, Fk = self.predict(x_current[i-1], x_update[i-1], P_update[i-1])
                F.append(Fk)
                
            x_predict.append(x.reshape((self.n,1)))
            P_predict.append(P)
            measure = self.y_list[i]
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
    
    def cost(self, xk):
        sum1 = 0
        sum2 = 0
        for i in range(len(self.y_list)):
            zk = self.y_list[i]
            hk = self.env.measurement(xk[i])
            sum1 += np.transpose((zk-hk)) @ np.linalg.inv(self.R) @ (zk-hk)
        for i in range(len(self.y_list)-1):
            xnext = xk[i+1]
            fk = self.env.nextstate(xk[i])
            sum2 += np.transpose(xnext-fk) @ np.linalg.inv(self.Q) @ (xnext-fk)
            
        L_x1 = np.transpose(xk[0] - self.x0_hat) @ np.linalg.inv(self.P) @ (xk[0] - self.x0_hat) + sum1 + sum2
        return L_x1

    def directional_derivative(self, xk, del_x):
        sum1 = 0
        sum2 = 0
        for i in range(len(del_x)-1):
            f = self.env.nextstate(xk[i])
            F = self.env.Jacobian(xk[i])
            a = del_x[i+1] - F @ del_x[i]
            b = np.linalg.inv(self.Q) @ (xk[i+1] - f)
            sum1 += np.transpose(a) @ b
            
        for i in range(len(del_x)):
            zk = self.y_list[i]
            hk = self.env.measurement(xk[i])
            H = self.env.JacobianMeasure(xk[i])
            c = H @ del_x[i]
            sum2 += np.transpose(c) @ np.linalg.inv(self.R) @ (zk - hk)
        
        d = 2 * np.transpose(del_x[0]) @ np.linalg.inv(self.P) @ (xk[0] - self.x0_hat) + 2 * sum1 - 2 * sum2
        return d


    def solve(self, x_current):
        c1 = 1e-4
        tau = 1 / 2
        x_i = x_current
        grad_norm = 10
        cost1 = self.cost(x_i)
        print('first cost = ', cost1)
        while grad_norm >= 1e-6:
            
            x_predict, x_i1 = self.step(x_i)
            del_x = np.array(x_i1) - np.array(x_i)
            print('delx shape = ',del_x.shape)
            alpha = 1
            d = self.directional_derivative(x_i, del_x)
            grad_norm = np.linalg.norm(d)
            print('grad_norm =', grad_norm)
            
            xk = x_i + alpha*del_x 
            cost1 = self.cost(xk)
            cost2 = self.cost(x_i)
            
            
            count = 0 
            while cost1 >= cost2 and count < 20:
                print('cost candidate = ', cost1)
                alpha *= tau
                xk = x_i + alpha*del_x 
                cost1 = self.cost(xk)
                count += 1
            
            if cost1 > cost2:
                break
            x_i = x_i + alpha*del_x
            print('Accepted Cost = ', cost1)
        if grad_norm >= 1e-6:
            print('DID NOT CONVERGE')
            
        return x_i

        


