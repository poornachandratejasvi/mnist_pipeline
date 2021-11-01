#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import requests
import tarfile
import os.path


# In[2]:


model_path = os.environ.get("model_path", "/tmp/pipeline/model/")
etl_data_path = os.environ.get("etl_path", "/tmp/pipeline/etl/")
model_store = os.environ.get("model_store_path", "/tmp/pipeline/rdmodelstore/")
min_fscore = os.environ.get("min_f1_score", "0.90")


# In[3]:


models=[]
for root, dirs, files in os.walk(model_path):
    print("model location:",root)
    print("list of models: ",dirs)
    models=dirs
    break


# In[4]:


accepcetd_model=""
greater_model_f1=0
for mod in models:
    print("\n#########  model name: ",mod,"#########\n")
    model = tf.keras.models.load_model(model_path+"/"+mod)
    print(model.summary())
    with np.load(etl_data_path+"testing_data.npz") as data:
        x_test = data['imgs']
        y_test = data['labels']
    x_test = np.expand_dims(x_test, -1)
    
    predictions = model.predict(x_test)
    print(classification_report(y_test, np.argmax(predictions,axis=1)))
    model_f1=f1_score(y_test, np.argmax(predictions,axis=1),average='macro')
#     model_f1=round(model_f1, 2)
    print("model " +mod+ "f1 score: ",model_f1)
    if (model_f1 >= float(min_fscore)) :
        print("model " +mod+ " meets minimum eligible F1 score")
        if model_f1 > greater_model_f1:
            greater_model_f1= model_f1
            accepcetd_model= mod
    else:
        print("no model is elligible")
print("selected model for deployment ",accepcetd_model)


# In[5]:


if not os.path.exists(model_store):
    os.makedirs(model_store)


# In[6]:



output_filename=model_store+"/modeltoupload.tar"
source_dir=model_path+"/"+accepcetd_model
def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname="model")
make_tarfile(output_filename,source_dir)


# In[7]:


payload = {}
url="http://localhost:8082/uploadmodel"
files = [('files', open(output_filename,'rb'))]
headers = {'Content-Transfer-Encoding': 'application/gzip'}
response = requests.request("POST", url, headers=headers, data = payload, files = files, verify = False)


# In[8]:


print(response.text)


# In[ ]:





# In[ ]:




