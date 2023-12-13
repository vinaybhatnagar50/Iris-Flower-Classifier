#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics  import f1_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')


# In[2]:


iris =pd.read_csv('C:\\Users\\ok\\Downloads\\Iris.csv')


# In[3]:


iris.head()


# In[4]:


iris.nunique()


# In[8]:


iris .isnull().sum()


# In[9]:


iris.info()


# In[10]:


iris.describe()


# In[12]:


iris.shape


# In[13]:


train=iris.copy()
train


# In[14]:


train=train.drop(['Id'],axis=1)
b=(train.columns)
train.shape


# In[15]:


b


# In[16]:


train['Species']=pd.factorize(train['Species'])[0]
train


# In[17]:


sns.distplot(train['Species'], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 3})
plt.xlabel("count")
plt.ylabel("Density")
plt.title("Density of price")
plt.legend("Price")
plt.show()


# In[18]:


y = train.Species
train.drop(['Species'], axis=1, inplace=True)
x = train


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[20]:


lr = LogisticRegression()
lr.fit(X_train,y_train)

rf = RandomForestClassifier()
rf.fit(X_train,y_train)


gbr = GradientBoostingClassifier()
gbr.fit(X_train,y_train)

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)


# In[21]:


y_pred1 = lr.predict(X_test)
y_pred2 = rf.predict(X_test)
y_pred3 = gbr.predict(X_test)
y_pred4 = dt.predict(X_test)


# In[22]:


accuracylr = accuracy_score(y_test, y_pred1)
print("Accuracy: ", accuracylr)

accuracyrf = accuracy_score(y_test, y_pred2)
print("Accuracy: ", accuracyrf)

accuracygbr = accuracy_score(y_test, y_pred3)
print("Accuracy: ", accuracygbr)

accuracydt = accuracy_score(y_test, y_pred4)
print("Accuracy: ", accuracydt)


# In[23]:


import seaborn as sns
sns.pairplot(iris)


# In[24]:


import joblib
joblib.dump(dt, "IrisFlowerClassification.pkl")


# In[ ]:




