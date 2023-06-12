#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[2]:


data = pd.read_csv('train.csv')


# In[3]:


data.info()


# In[4]:


data


# In[5]:


data.isna().sum()


# In[6]:


data.describe()


# In[7]:


data.shape


# In[8]:


data.dtypes


# In[9]:


train_data, test_data, train_labels, test_labels = train_test_split(
    data['id'], data['target'], test_size=0.2, random_state=42)


# In[10]:


vectorizer = TfidfVectorizer()


# In[11]:


train_features = vectorizer.fit_transform(train_data)
test_features = vectorizer.transform(test_data)


# In[12]:


model = LinearRegression()
model.fit(train_features, train_labels)


# In[13]:


predictions = model.predict(test_features)


# In[14]:


mse = mean_squared_error(test_labels, predictions)
print("Mean Squared Error:", mse)

