#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import os
from PIL import Image


# In[2]:


output_path = os.environ.get("etl_path", "/tmp/production/etl/")
input_path = os.environ.get("sensor_img_path", "/tmp/production/sensor_image/")


# In[3]:


if not os.path.exists(output_path):
    os.makedirs(output_path)


# # Load image

# In[4]:


file="predict.png"
single_im = Image.open(input_path+file)
single_img = np.array(single_im)
print(single_img.shape)
single_img = single_img.reshape(1,28,28,1)
single_img = tf.keras.utils.normalize(single_img, axis=1)


# In[ ]:





# In[5]:


print(single_img.shape)


# In[6]:


np.savez_compressed(output_path+"to_predict.npz", imgs=single_img)


# In[ ]:





# In[ ]:




