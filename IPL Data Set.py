#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("Ipl matches 08-20.csv")


# In[4]:


df.shape


# In[5]:


df.head()


# In[10]:


df.info()


# In[11]:


df.describe()


# In[9]:


df.venue.value_counts()


# In[12]:


df.head(5)


# In[16]:


from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
df["city"] = number.fit_transform(df["city"].astype("str"))
df["venue"] = number.fit_transform(df["venue"].astype("str"))
df["player_of_match"] = number.fit_transform(df["player_of_match"].astype("str"))
df["team1"] = number.fit_transform(df["team1"].astype("str"))
df["team2"] = number.fit_transform(df["team2"].astype("str"))
df["toss_winner"] = number.fit_transform(df["toss_winner"].astype("str"))
df["toss_decision"] = number.fit_transform(df["toss_decision"].astype("str"))
df["winner"] = number.fit_transform(df["winner"].astype("str"))
df["result"] = number.fit_transform(df["result"].astype("str"))
df["result_margin"] = number.fit_transform(df["result_margin"].astype("str"))
df["eliminator"] = number.fit_transform(df["eliminator"].astype("str"))
df["method"] = number.fit_transform(df["method"].astype("str"))
df["umpire1"] = number.fit_transform(df["umpire1"].astype("str"))
df["umpire2"] = number.fit_transform(df["umpire2"].astype("str"))


# In[17]:


df.head(10)


# In[28]:


df.describe()


# In[22]:


X = df[["neutral_venue","result_margin"]]
y = df["result"]


# In[40]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.01)


# In[41]:


X_train.shape,X_test.shape


# In[42]:


from sklearn.tree import DecisionTreeClassifier,plot_tree
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)


# In[43]:


plt.figure(figsize=[10,20])
_ = plot_tree(clf,feature_names=["neutral_venue","result_margin"])


# In[44]:


y_predict = clf.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_test,y_predict),accuracy_score(y_test,y_predict)


# In[ ]:




