#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import os
import json
import requests


# In[2]:


etl_data_path = os.environ.get("etl_path", "/tmp/production/etl/")


# In[3]:


with np.load(etl_data_path+"to_predict.npz") as data:
    x_test = data['imgs']


# In[4]:


type(x_test)


# In[5]:


payload=json.dumps(x_test.tolist())


# In[6]:


# payload = {}
url="http://localhost:8082/predict"
headers = {'Content-type': 'application/json'}
response = requests.request("POST", url, headers=headers, data = payload, verify = False)


# In[7]:


response


# In[8]:


response.text


# In[ ]:




