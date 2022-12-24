import time
import numpy as np
import scipy.linalg
import robot_visualizer
import FOR_Functions as forf
import matplotlib.pyplot as plt

T = 10.
end_effector_initial = np.array([[0], [0],[1.301]])
end_effector_goal1 = np.array([[0.7], [0.2],[0.7]])
end_effector_goal2 = np.array([[0.3], [0.5],[0.9]])
ee_pos_1 = 0
zeros = np.array([[0],[0],[0],[0],[0],[0],[0]])
ee_pos_1 = 0

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


def robot_controller3(t, joint_positions, joint_velocities):
    """A typical robot controller
        at every time t, this controller is called by the simulator. It receives as input
        the current joint positions and velocities and needs to return a [7,1] vector
        of desired torque commands

        As an example, the current controller implements a PD controller and at time = 5s
        it makes joint 2 and 3 follow sine curves
    """

    desired_joint_positions = np.zeros([7,1])
    desired_joint_velocities = np.zeros([7,1])

    # here we will only use the D controller to inject small joint damping
    D = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    #D = np.array([0, 0, 0, 0, 0, 0, 0])
    Dn = 15
    K = 5
    ##TODO - implement gravity compensation and impedance control
    Gravity = robot_visualizer.rnea(joint_positions,zeros,zeros)

    if t < 5.:
        des_ee_positions, des_ee_velocities = forf.get_point_to_point_motion(end_effector_initial,end_effector_goal1,t,5)

    else:
        des_ee_positions, des_ee_velocities = forf.get_point_to_point_motion(end_effector_goal1,end_effector_goal2,t-5,5)

    I = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
    f = forf.forward_kinematics(joint_positions)
    ee_pos = f[[0,1,2],3].reshape((-1,1))
    T_actual = np.column_stack((I,f[:,3]))
    T_desired = np.column_stack((np.identity(3),des_ee_positions))
    T_desired = np.vstack((T_desired,np.array([0,0,0,1])))
    Js = forf.get_space_jacobian(joint_positions)
    #print('Js = ', Js)
    #print('T_actual = ', T_actual)
    Jse = np.dot(forf.getAdjoint(scipy.linalg.inv(T_actual)), Js)
    #print('Jse = ', Jse)
    V = Jse @ joint_velocities
    Jse_translation = Jse[[3,4,5],:]
    JT = np.transpose(Jse_translation)
    Pseudo_J = forf.pinv(Jse_translation)
    V_translation = V[[3,4,5],:]



    desired_joint_torques = Gravity + JT @ (K * (des_ee_positions - ee_pos) + Dn * (des_ee_velocities - V_translation)) - (np.diag(D) @(joint_velocities))

    desired_joint_velocities = Pseudo_J @ des_ee_velocities
    #print(des_ee_velocities)
    global ee_pos_1
    desired_joint_positions = ee_pos_1 + (0.001)*desired_joint_velocities
    ee_pos_1 = desired_joint_positions
    ee_velocities = Jse_translation @ joint_velocities
    ee_positions = ee_pos

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
robot_visualizer.display_ball([0.7, 0.2,0.7])
robot_visualizer.display_ball2([0.3, 0.5,0.9])
robot_visualizer.simulate_robot(robot_controller3, T=T, gravity_comp = False)
