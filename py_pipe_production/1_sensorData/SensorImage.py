#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import os


# In[2]:


output_path = os.environ.get("etl_path", "/tmp/production/sensor_image/")
base_path = os.environ.get("saved_data_dir", "/tmp/production/data/mnist/")


# In[3]:


if not os.path.exists(output_path):
    os.makedirs(output_path)
    
if not os.path.exists(base_path):
    os.makedirs(base_path)


# In[4]:


DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
tf.keras.utils.get_file(base_path+'mnist.npz', DATA_URL)


# # Load from .npz file

# In[5]:


with np.load(base_path+"mnist.npz") as data:
    x_test = data['x_train'][59000:] #use 10k for production


# In[10]:


len(x_test)
from random import randrange
number=randrange(len(x_test))
print(number)


# In[11]:


# In[12]:


from PIL import Image
im = Image.fromarray(x_test[number])
im.save(output_path+"/predict.png")


# In[ ]:




