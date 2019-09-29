
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import seaborn as sn


from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import statsmodels.api as sm

#from sklearn.matrics import r2_score


# In[ ]:

A=pd.read_csv('/Users/ranvir/Desktop/50_Startups.csv')
A.head()


# In[ ]:

#EDA : to know/calculate the  dependency of all variable on each other
sn.pairplot(A)


# In[ ]:

#to plot the graph for EDA variable
plt.show()


# In[ ]:

A.corr()


# In[ ]:

X=A[['RND','MKT']]
Y=A[['PROFIT']]
xtrain,xtest,ytrain,ytest= train_test_split(X,Y,test_size=0.20,random_state=40)


# In[ ]:

lm=LinearRegression()
model=lm.fit(xtrain,ytrain)


# In[ ]:

model


# In[ ]:



