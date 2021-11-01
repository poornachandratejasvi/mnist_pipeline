#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import os


# In[2]:


output_path = os.environ.get("etl_path", "/tmp/pipeline/etl/")
base_path = os.environ.get("saved_data_dir", "/tmp/pipeline/data/mnist/")
modelAppend = os.environ.get("model_append", "False")


# In[3]:


if not os.path.exists(output_path):
    os.makedirs(output_path)


# # Load from .npz file

# In[4]:


if modelAppend.lower() != "true" :
    with np.load(base_path+"mnist.npz") as data:
        x_train = data['x_train'][:59000] #going to use 10k for evalution
        y_train = data['y_train'][:59000]
        x_test = data['x_test']
        y_test = data['y_test']
else:
    with np.load(base_path+"mnist.npz") as data:
        x_train = data['x_train'][59000:] #for CT
        y_train = data['y_train'][59000:]
        x_test = data['x_test']
        y_test = data['y_test']


# In[5]:


print("size of training data",len(x_train))


# # scales data between 0 and 1

# In[6]:


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# In[7]:


np.savez_compressed(os.path.join(output_path, "training_data"), imgs=x_train, labels=y_train)


# In[8]:


np.savez_compressed(os.path.join(output_path, "testing_data"), imgs=x_test, labels=y_test)


# In[ ]:





# In[ ]:




