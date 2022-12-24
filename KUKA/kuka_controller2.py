import time
import numpy as np
import scipy.linalg
import robot_visualizer
import FOR_Functions as forf
import matplotlib.pyplot as plt

with open('desired_end_effector_positions.npy', 'rb') as f:
    p = np.load(f)
theta = np.array([[0],[0],[0],[0],[0],[0],[0]])
theta_1 = np.array([[1],[1],[-1],[-1],[1],[1],[1]])

T = 10.
end_effector_initial = np.array([[0], [0],[1.301]])
end_effector_goal1 = np.array([[0.7], [0.2],[0.7]])
end_effector_goal2 = np.array([[0.3], [0.5],[0.9]])
P = 15
ee_pos_1 = 0

#null term
x1 = 10**-15
x2 = 10**-15
x3 = 10**-15
x4 = 10**-15
x5 = 10**-15
x6 = 10**-15
x7 = 10**-15
null_term = np.array([[x1],[x2],[x3],[x4],[x5],[x6],[x7]])

## this code is to save what the controller is doing for plotting and analysis after the simulation
global save_joint_positions, save_joint_velocities, save_t, ind
global save_des_joint_positions, save_des_joint_velocities
global save_ee_positions, save_ee_velocities
save_joint_positions = np.zeros([7,int(np.ceil(T / 0.001))+1])
save_joint_velocities = np.zeros_like(save_joint_positions)
save_des_joint_positions = np.zeros_like(save_joint_positions)
save_des_joint_velocities = np.zeros_like(save_joint_positions)
save_ee_positions = np.zeros([3,int(np.ceil(T / 0.001))+1])
save_ee_velocities = np.zeros([3,int(np.ceil(T / 0.001))+1])
save_t = np.zeros([int(np.ceil(T / 0.001))+1])
ind=0
# end of saving code


def robot_controller2(t, joint_positions, joint_velocities):
    """A typical robot controller
        at every time t, this controller is called by the simulator. It receives as input
        the current joint positions and velocities and needs to return a [7,1] vector
        of desired torque commands

        As an example, the current controller implements a PD controller and at time = 5s
        it makes joint 2 and 3 follow sine curves
    """

    desired_joint_positions = np.zeros([7,1])
    desired_joint_velocities = np.zeros([7,1])

    # here we will only use a D controller (i.e. on the desired joint velocities)
    # we increased the D gain for that purpose compared to the previous controller
    D = np.array([4.,4,4,4,4,4,4.])


    ##TODO - find the desired joint velocities
    if t < 5.:
        des_ee_positions, des_ee_velocities = forf.get_point_to_point_motion(end_effector_initial,end_effector_goal1,t,5)

    else:
        des_ee_positions, des_ee_velocities = forf.get_point_to_point_motion(end_effector_goal1,end_effector_goal2,t-5,5)

    I = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
    f = forf.forward_kinematics(joint_positions)
    ee_pos = f[[0,1,2],3].reshape((-1,1))
    T = np.column_stack((I,f[:,3]))
    Js = forf.get_space_jacobian(joint_positions)
    Jse = forf.getAdjoint(scipy.linalg.inv(T)) @ Js
    Jse_translation = Jse[[3,4,5],:]
    Pseudo_J = forf.pinv(Jse_translation)
    #print(f)
    #print(current_ee_pos)
    #print(T)
    #time.sleep(5)


    desired_joint_velocities = Pseudo_J @ (P * (des_ee_positions - ee_pos) + des_ee_velocities) + (np.identity(7) - Pseudo_J @ Jse_translation) @ null_term
    global ee_pos_1
    desired_joint_positions = ee_pos_1 + (0.001)*desired_joint_velocities
    ee_pos_1 = desired_joint_positions
    ee_velocities = Jse_translation @ joint_velocities
    ee_positions = ee_pos

    desired_joint_torques = np.diag(D) @ (desired_joint_velocities - joint_velocities)


    ## this code is to save what the controller is doing for plotting and analysis after the simulation
    global save_joint_positions, save_joint_velocities, save_t, ind
    global save_des_joint_positions, save_des_joint_velocities
    save_joint_positions[:,ind] = joint_positions[:,0]
    save_joint_velocities[:,ind] = joint_velocities[:,0]
    save_des_joint_positions[:,ind] = desired_joint_positions[:,0]
    save_des_joint_velocities[:,ind] = desired_joint_velocities[:,0]
    save_ee_positions[:, ind] = ee_positions[:, 0]
    save_ee_velocities[:, ind] = ee_velocities[:, 0]
    save_t[ind] = t
    ind += 1
    ## end of saving code

    return desired_joint_torques

robot_visualizer.start_robot_visualizer()
robot_visualizer.display_ball(end_effector_goal1[:,0])
robot_visualizer.display_ball2(end_effector_goal2[:,0])
robot_visualizer.simulate_robot(robot_controller2, T=T)
