#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def est(x,y):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
    SS_xy = np.sum(y*x) - n*m_x*m_y
    SS_xx = np.sum(x*x) - n*m_x*m_x
    b_1 = SS_xy/SS_xx
    b_0 = m_y - b_1*m_x
    return(b_0,b_1)


# In[6]:


def plot(x,y,b):
    plt.scatter(x,y)
    y_p = b[0] + b[1]*x
    plt.plot(x,y_p)
    plt.xlabel('X')
    plt.ylabel('Y')


# In[4]:


x = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([1,3,2,5,7,8,8,9,10,12])
b = est(x,y)
print(b)


# In[7]:


plot(x,y,b)


# In[ ]:




