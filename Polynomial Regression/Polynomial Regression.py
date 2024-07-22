#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dt = pd.read_csv('Position_Salaries.csv')
dt


# In[4]:


x = dt.iloc[:,1:2].values
y = dt.iloc[:,2].values


# In[5]:


from sklearn.linear_model import LinearRegression
lin_reg =LinearRegression()
lin_reg.fit(x,y)


# In[7]:


from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree=2)
x_pol = pol_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_pol,y)


# In[8]:


pol_reg3 = PolynomialFeatures(degree=3)
x_pol3 = pol_reg3.fit_transform(x)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_pol3,y)


# In[9]:


plt.scatter(x,y)
plt.plot(x,lin_reg.predict(x),color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# In[14]:


plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(pol_reg.fit_transform(x)),color='blue')
plt.plot(x,lin_reg3.predict(pol_reg3.fit_transform(x)),color='green')
plt.xlabel('X')
plt.ylabel('y')
plt.show()


# In[15]:


X_grid=np.arange(min(x),max(x),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(x,y,color='red')
plt.plot(X_grid,lin_reg3.predict(pol_reg3.fit_transform(X_grid)),color='blue')
plt.show()


# In[ ]:




