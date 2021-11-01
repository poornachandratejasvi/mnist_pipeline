#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import os


# In[2]:


base_path = os.environ.get("saved_data_path", "/tmp/pipeline/data/mnist/")


# In[3]:


if not os.path.exists(base_path):
    os.makedirs(base_path)


# In[4]:


DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
tf.keras.utils.get_file(base_path+'mnist.npz', DATA_URL)


# In[ ]:




