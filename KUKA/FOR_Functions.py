import scipy.linalg
import numpy as np

with open('desired_end_effector_positions.npy', 'rb') as f:
    p = np.load(f)

def vec_to_skew(x):
  x = np.array([[0, -x[2][0], x[1][0]],
                  [x[2][0], 0, -x[0][0]],
                  [-x[1][0], x[0][0], 0]])

  return x

def twist_to_skew(V): # ex. S to [S]

    w = np.array([V[0],V[1],V[2]])

    v = np.array([V[3],V[4],V[5]])

    w = vec_to_skew(w)
    zeros = np.array([0, 0, 0, 0])

    V_Bracket = np.column_stack((w, v))
    V_Bracket = np.vstack([V_Bracket, zeros])


    return V_Bracket;

def exp_twist_bracket(V):
  VBracket = twist_to_skew(V)
  T = scipy.linalg.expm(VBracket)

  return T

def inverseT(T):
  T = np.linalg.inv(T)
  return T

def T_Rp(T):

    R = np.array([[T[0][0],T[0][1],T[0][2]],[T[1][0],T[1][1],T[1][2]],[T[2][0],T[2][1],T[2][2]]])

    p = np.array([[T[0][3]],[T[1][3]],[T[2][3]]])

    return R, p

def getAdjoint(T):
    R, p = T_Rp(T)
    zeros = np.zeros((3, 3))
    p_skew = vec_to_skew(p)

    pR = p_skew @ R

    A1 = np.column_stack((R, zeros))
    A2 = np.column_stack((pR, R))
    AdT = np.vstack([A1, A2])

    return AdT

def forward_kinematics(theta):
    S = np.array([[0,0,0,0,0,0,0],
              [0,1,0,-1,0,1,0],
              [1,0,1,0,1,0,1],
              [0,-.36,0,.78,0,-1.18,0],
              [0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0]])

    M = np.array([[1,0,0,0,],
              [0,1,0,0],
              [0,0,1,1.301],
              [0,0,0,1]])
    n = 0
    m = len(S[0])
    eS_Total = np.identity(4)
    if len(theta[0]) > 1:
        theta = theta.reshape(-1,1)
    for i in range(m):
        Sn = S[:, n]
        Sn = Sn.reshape(-1, 1)
        Pose_Sn = twist_to_skew(Sn)
        #print(theta[n,0])
        eS = scipy.linalg.expm(Pose_Sn*theta[n, 0])
        eS_Total = eS_Total @ eS
        n = n+1
    T = eS_Total @ M
    return T

def get_space_jacobian(theta): # Spatial Jacobian
    S = np.array([[0,0,0,0,0,0,0],
              [0,1,0,-1,0,1,0],
              [1,0,1,0,1,0,1],
              [0,-.36,0,.78,0,-1.18,0],
              [0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0]])
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

def pinv(J, alpha = 10**-5):
    JT = np.transpose(J)
    Psuedo_J = JT @ np.linalg.inv((J @ JT + alpha*np.identity(np.size(J, 0))))
    return Psuedo_J

def compute_IK_position(theta,p_des,error=.001):
    singularity = []
    for j in range(len(p_des[0])):
        p_desired = p_des[:,j].reshape((-1,1))
        i = 0
        e = error + 1

        while (np.linalg.norm(e) >= error) and (i < 500):
            I = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
            T_fk = forward_kinematics(theta)
            p_current = T_fk[(0,1,2),3].reshape((-1,1))
            Js = get_space_jacobian(theta)
            Tsb = np.column_stack((I,T_fk[:,3]))
            Jsb = getAdjoint(np.linalg.inv(Tsb)) @ Js
            j_trans = Jsb[(3,4,5),:]
            Jpinv = scipy.linalg.pinv(j_trans)

            e = Jpinv @ (p_desired - p_current)
            theta = theta + e
            i = i+1
            #print(theta)

        if j == 0:
            theta_des = theta
        else:
            theta_des = np.hstack((theta_des,theta))
        if i == 500:# and (np.linalg.norm(e) <= .1):
            singularity.append(j)
            print("Error in positions Singularities", j+1, ' =  ')
            print(e)

    return theta_des, singularity

def compute_IK_position_nullspace(theta,theta_1,p_des,error=.001):
    singularity = []
    for j in range(len(p_des[0])):
        p_desired = p_des[:,j].reshape((-1,1))
        i = 0
        e = error + 1

        while (np.linalg.norm(e) >= error) and (i < 500):
            I = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
            T_fk = forward_kinematics(theta)
            p_current = T_fk[(0,1,2),3].reshape((-1,1))
            Js = get_space_jacobian(theta)
            Tsb = np.column_stack((I,T_fk[:,3]))
            Jsb = getAdjoint(np.linalg.inv(Tsb)) @ Js
            j_trans = Jsb[(3,4,5),:]
            Jpinv = scipy.linalg.pinv(j_trans)

            e = Jpinv @ (p_desired - p_current) + ((np.identity(7) - (Jpinv @ j_trans)) @ (theta_1 - theta))
            theta = theta + e
            i = i+1
            #print(theta)
            #print(Jpinv)
        if j == 0:
            theta_des = theta
        else:
            theta_des = np.hstack((theta_des,theta))
        if i == 500:
            singularity.append(j)
    return theta_des, singularity 

def get_point_to_point_motion(theta_init,theta_des,t,T): #theta_desired="no"

    pos = theta_init + ((10*(t**3)/(T**3))+(-15*(t**4)/(T**4))+(6*(t**5)/(T**5))) * (theta_des - theta_init)
    vel = ((30*(t**2)/(T**3))+(-60*(t**3)/(T**4))+(30*(t**4)/(T**5))) * (theta_des - theta_init)

    return pos, vel
