#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[2]:


data = pd.read_csv('spam .csv',encoding='latin-1')


# In[3]:


data


# In[4]:


data.info


# In[5]:


data.describe()


# In[6]:


data.shape


# In[7]:


data.dtypes


# In[8]:


data.isna().sum()


# In[9]:


"Number of messages :",len(data)


# In[10]:


"Columns:" , data.columns 


# In[11]:


train_data, test_data, train_labels, test_labels = train_test_split(
    data['v1'], data['v2'], test_size=0.2, random_state=42)


# In[12]:


vectorizer = TfidfVectorizer()


# In[13]:


train_features = vectorizer.fit_transform(train_data)
test_features = vectorizer.transform(test_data)
data


# In[14]:


classifier = SVC()
classifier.fit(train_features, train_labels)


# In[17]:


predictions = classifier.predict(test_features)


# In[20]:


data=data.rename(columns={"v1":"target","v2":"text"},inplace=False)


# In[23]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[24]:


data['target']=encoder.fit_transform(data['target'])


# In[35]:


sns.heatmap(data.corr(),annot=True,cmap='coolwarm')


# In[37]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
gnb=GaussianNB()
mlnb=MultinomialNB()
bnb=BernoulliNB()


# In[39]:


train_data, test_data, train_labels, test_labels = train_test_split(
    data['text'], data['target'], test_size=0.2, random_state=42)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




