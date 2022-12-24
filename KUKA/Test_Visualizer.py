import numpy as np
import robot_visualizer
import time
import matplotlib.pyplot as plt

robot_visualizer.start_robot_visualizer()
'''
q = np.random.sample([7])
print(f'we show the configuration for the angles {q}')
robot_visualizer.display_robot(q)
'''
# load the file
with open('desired_end_effector_positions.npy', 'rb') as f:
    desired_endeff = np.load(f)

# first we display the robot in 0 positionVisualize_KUKA.py
robot_visualizer.display_robot(np.zeros(7))

# for each end-eff position
for i in range(desired_endeff.shape[1]):
    # displays the desired endeff position
    robot_visualizer.display_ball(desired_endeff[:,i])
    time.sleep(1.)
