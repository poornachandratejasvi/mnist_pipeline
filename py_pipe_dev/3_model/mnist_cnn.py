#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
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


# In[4]:


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
# x_train = x_train.astype("float32") / 255
# x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
# print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)


# ## Build the model

# In[5]:


model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()


# In[6]:


model_file_path=model_path+"/poorna_cnn_mnist.model"
if modelAppend.lower() == "true" and os.path.exists(model_file_path) == True:
    print("going to use existing model")
    model = keras.models.load_model(model_file_path)
    model.summary()


# ## Train the model

# In[7]:


batch_size = batch
epochs = epoch
opt = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=opt,  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

checkpoint = ModelCheckpoint(filepath=model_file_path,save_best_only=True,monitor='loss',verbose=1,mode='min')

early_stop = EarlyStopping(monitor = "loss", patience = 10, verbose = 1)


model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True,verbose=1,callbacks=[early_stop, checkpoint], validation_split=0.1)


# In[8]:


# model.save(model_file_path)


# In[ ]:





# In[ ]:




