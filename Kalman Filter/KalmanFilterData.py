# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:37:30 2022

@author: reimoj
"""
import numpy as np
import scipy.io as sio

u = sio.loadmat('Accel_Data')
z = sio.loadmat('Vel_Data')

u = u['a']
z = np.array(z['v'])
print('u = ',z)