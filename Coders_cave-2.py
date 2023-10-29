#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import string
import re


# In[2]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[3]:


data=pd.read_csv("emails.csv")


# In[4]:


data


# In[5]:


# Dropijng any null values
data["spam"].value_counts()


# In[6]:


print(f"Number of missing data : \n{data.isnull().sum()}\n")


# In[7]:


X = data["text"]
y = data["spam"]


# In[8]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[9]:


data_list = []
# iterating through all the text
for text in X:
    text = re.sub(r"[!@#$(),n%^*?:;~`0-9]", ' ', str(text))
    text = re.sub(r'[[]]', ' ', str(text))
    # converting the text to lower case
    text = text.lower()
    # appending to data_list
    data_list.append(str(text))


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(data_list).toarray()
X.shape


# In[11]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# In[12]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)


# In[13]:


y_pred = model.predict(x_test)


# In[14]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy is :",ac)


# In[ ]:




