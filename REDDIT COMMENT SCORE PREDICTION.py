#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("Train_Data.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.head()


# In[6]:


df['score vs. votes'] = df['parent_score']==df['parent_votes']
df['score vs. votes'].nunique()


# In[7]:


df.drop(['parent_votes', 'score vs. votes'], axis= 1, inplace=True)
df.head()


# In[8]:


cor = df.corr()
sns.heatmap(cor)


# In[9]:


categorical_cols = ['text','author','parent_text','parent_author']
for col in df[categorical_cols]:
    df[col] = df[col].str.lower()
    df[col] = df[col].str.strip()
    df.head()
    


# In[10]:


df.head()


# In[11]:


import string

for col in df[categorical_cols]:
    df[col] = df[col].apply(lambda x:''.join([i for i in x if i not in string.punctuation]))
df.head()


# In[12]:


sns.countplot(df['controversiality'])


# In[13]:


df.isnull().sum()


# In[14]:


train_qs = pd.Series(df['author'].tolist()).astype(str)


# In[15]:


train_qs2 = pd.Series(df['parent_author'].tolist()).astype(str)

