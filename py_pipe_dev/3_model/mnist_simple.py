#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import json


# In[2]:


model_path = os.environ.get("model_path", "/tmp/pipeline/model/")
etl_data_path = os.environ.get("etl_path", "/tmp/pipeline/etl/")
epoch = int(os.environ.get("epochs", "3"))
batch = int(os.environ.get("batch_size", "10"))
learning_rate = float(os.environ.get("learning_rate", "0.01"))
modelAppend = os.environ.get("model_append", "False")


# In[3]:


with np.load(etl_data_path+"training_data.npz") as data:
    x_train = data['imgs']
    y_train = data['labels']
x_train = np.expand_dims(x_train, -1)


# In[4]:


model = tf.keras.models.Sequential(name="mnist_without_CNN")  # a basic feed-forward model
model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for probability distribution


# In[5]:


model_file_path=model_path+"/poorna_mnist.model"
if modelAppend.lower() == "true" and os.path.exists(model_file_path) == True:
    print("going to use existing model")
    model = tf.keras.models.load_model(model_file_path)


# In[6]:


opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=opt,  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track
checkpoint = ModelCheckpoint(filepath=model_file_path,save_best_only=True,monitor='val_loss',verbose=1,mode='min')

early_stop = EarlyStopping(monitor = "val_loss", patience = 10, verbose = 1)


# In[7]:



model.fit(x_train, y_train, epochs=epoch,batch_size=batch,shuffle=True,verbose=1,callbacks=[early_stop, checkpoint], validation_split=0.1)  # train the model


# In[8]:


model.summary()


# In[9]:


model.save(model_file_path)


# In[10]:


# resultDict={}

# resultDict["model_type_CNN"]="False"

# with open(model_path+'/result.json', 'w') as fp:
#     json.dump(resultDict, fp)


# In[ ]:




