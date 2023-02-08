"""
This is the enviorment file for the kalman filter with smoothing. 
This file contains all the necessesary files to model a pendulum using a Kalman Filter. 
"""
import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt

class pendulumn:
    def __init__ (self, m, l, H, umax, theta_des, dt):
        self.dt = dt
        self.g = 9.81
        self.l = l
        self.m = 1
        self.H = H
        self.umax = umax
        self.theta_des = theta_des
        self.dim_x = 2
    

    def vec_to_skew(self, x):
        x = np.array([[0, -x[2][0], x[1][0]],
                      [x[2][0], 0, -x[0][0]],
                      [-x[1][0], x[0][0], 0]])

       return x

    def twist_to_skew(self, V): # ex. S to [S]
    
        w = np.array([V[0],V[1],V[2]])
   
        v = np.array([V[3],V[4],V[5]])

        w = vec_to_skew(w)
        zeros = np.array([0, 0, 0, 0])

        V_Bracket = np.column_stack((w, v))
        V_Bracket = np.vstack([V_Bracket, zeros])


        return V_Bracket;

    def exp_twist_bracket(self, V):
        VBracket = twist_to_skew(V)
        T = scipy.linalg.expm(VBracket)

      return T

    def inverseT(self, T):
        T = np.linalg.inv(T)
        return T
  
    def T_Rp(self, T):

        R = np.array([[T[0][0],T[0][1],T[0][2]],[T[1][0],T[1][1],T[1][2]],[T[2][0],T[2][1],T[2][2]]])

        p = np.array([[T[0][3]],[T[1][3]],[T[2][3]]])

        return R, p

    def getAdjoint(self, T):
        R, p = T_Rp(T)
        zeros = np.zeros((3, 3))
        p_skew = vec_to_skew(p)
        
        pR = p_skew @ R
        
        A1 = np.column_stack((R, zeros))
        A2 = np.column_stack((pR, R))
        AdT = np.vstack([A1, A2])
        
        return AdT

    def forward_kinematics(self, S,M,theta):
        n = 0
        m = len(S[0])
        eS_Total = np.identity(4)
        if len(theta[0]) > 1:
            theta = theta.reshape(-1,1)
        for i in range(m):
            Sn = S[:, n]
            Sn = Sn.reshape(-1, 1)
            Pose_Sn = twist_to_skew(Sn)
            eS = scipy.linalg.expm(Pose_Sn*theta[n, 0])
            eS_Total = eS_Total @ eS
            n = n+1
            T = eS_Total @ M
        return T

    def get_space_jacobian(self, S,theta): # Spatial Jacobian
        S0 = S[:,0]
        S0 = S0.reshape(-1,1)
        Js = S0
        n = 0
        m = len(S[0])-1
        Jp = 1
        e_S = np.identity(4)
        
        if len(theta[0]) > 1:
            theta = theta.reshape(-1,1)

        if len(S) > 1:
            for j in range(len(S[0])-1):
                Sn_1 = S[:, n + 1]
                Sn_1 = Sn_1.reshape(-1, 1)
                Sn = S[:, n]
                Sn = Sn.reshape(-1, 1)
                Pose_Sn = twist_to_skew(Sn)
                e_S1 = scipy.linalg.expm(Pose_Sn * theta[n, 0])
                e_S = e_S @ e_S1
                Jn = getAdjoint(e_S)
                J = Jn @ Sn_1
                Js = np.column_stack((Js, J))
                n = n+1
                m = m-1
                
        return Js

    def control(self, x):
        omega_des = 0
        Kp = 10
        Ki = 0.9
        
        u = -Kp*(x[0,0]-self.theta_des) - Ki*(x[1,0]-omega_des)
        U = np.array([[0],[u]])
        return u
    
    def nextstate(self, x, u=0):
        #print(u)
        #u = np.array([0,u]).reshape((2, 1))
        x_next = x[0,0] + self.dt*(x[1,0])
        v_next = (x[1,0] + self.dt*(u-self.g*np.sin(x[0,0]))) + u
        X_next = np.array([[x_next],[v_next]])
        return X_next
        
    def Jacobian(self, x):
        f = np.zeros((2,2))
        f[0,0] = 1
        f[0,1] = self.dt
        f[1,0] = -self.g*self.dt*np.cos(x[0])
        f[1,1] = 1
        
        return f
        
    def measurement(self, x):
        z = self.H @ x
        return z
    
    def JacobianMeasure(self, x):
        H_Jacobian = np.array([np.cos(x[0,0]),0]).reshape(1, 2)
        return self.H


