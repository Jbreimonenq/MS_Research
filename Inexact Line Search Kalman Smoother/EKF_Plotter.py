# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 23:07:18 2022

@author: reimoj
"""
import numpy as np
import matplotlib.pyplot as plt 

def plotIEKS(dt = None, m_value = None, p_value = None, a_value = None, state = None):
    m = None if m_value is None else True
    p = None if p_value is None else True
    a = None if a_value is None else True
    
    if m != None:
        m_value = np.stack(m_value, axis=0)
        plt.plot(np.dot(dt,range(len(m_value))), m_value[:,state], label = 'Measured Value')
    if p != None:
        p_value = np.stack(p_value, axis=0)
        plt.plot(np.dot(dt,range(len(p_value))), p_value[:,state], 'r--', label = 'Predicted Value')
    if a != None:
        a_value = np.stack(a_value, axis=0)
        plt.plot(np.dot(dt,range(len(a_value))), a_value[:,state], 'g', label = 'Actual Value')
    plt.title(f'State %d'% (state+1))
    plt.legend()
    #plt.xlim(2.5,3)
    #plt.ylim(-5,10)
    plt.grid()
    plt.show()
    
def subplotIEKS(axs, dt = None, m_value = None, p_value = None, a_value = None, state = None):
    m = None if m_value is None else True
    p = None if p_value is None else True
    a = None if a_value is None else True
    
    if m != None:
        m_value = np.stack(m_value, axis=0)
        axs.plot(np.dot(dt,range(len(m_value))), m_value[:,state], label = 'Measured Value')
    if p != None:
        p_value = np.stack(p_value, axis=0)
        axs.plot(np.dot(dt,range(len(p_value))), p_value[:,state], 'r--', label = 'Predicted Value')
    if a != None:
        a_value = np.stack(a_value, axis=0)
        axs.plot(np.dot(dt,range(len(a_value))), a_value[:,state], 'g', label = 'Actual Value')
    axs.grid()
    axs.legend()

    
def plotEKF(dt = None, m_value = None, p_value = None, a_value = None, state = None):
    m = None if m_value is None else True
    p = None if p_value is None else True
    a = None if a_value is None else True
    
    if m != None:
        m_value = np.stack(m_value, axis=0)
        plt.plot(np.dot(dt,range(len(m_value))), m_value[:,state], label = 'Measured Value')
    if p != None:
        p_value = np.stack(p_value, axis=0)
        plt.plot(np.dot(dt,range(len(p_value))), p_value[:,state], 'r--', label = 'Predicted Value')
    if a != None:
        a_value = np.stack(a_value, axis=0)
        plt.plot(np.dot(dt,range(len(a_value))), a_value[:,state], 'g', label = 'Actual Value')
    plt.title(f'State %d'% (state+1))
    plt.legend()
    plt.grid()
    plt.show()