# mnist_pipeline
mlops pipeline for mnist data

Architecture flow
-----------
Pipeline overview: 
![alt text](https://raw.githubusercontent.com/poornachandratejasvi/mnist_pipeline/main/picture/mnist-overall.png "pipeline overviw")

Pipeline execution flow: 
![alt text](https://raw.githubusercontent.com/poornachandratejasvi/mnist_pipeline/main/picture/mnist-detailed.png "execution flow")


Details
-----------
### model building pipeline
-----------

There are 4 steps in the model building or training phase

#### Get Data
In this stage we will download the mnist data from [mnist.npz](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz) and save the mnist.npz file

#### ETL

in this stage we will extract the data from mnist.npz file with split of data for training and testing , starting 59000 dataset is used for training, the range of pixel value is 0 to 255. Normalization of dataset is applied for making the pixel value to range between 0 to 1

#### model training

here were have created 2 model architecture, one with CNN and one without CNN, we traing both the model using the training data created in ETL stage, and save the model which is created in RD model store

we can increase the epos, batch_size and learning rate by seting the below parameter

env variable: **epochs** takes int value, batch_size takes int value, learning_rate takes float value

##### non CNN model achitecture

```
Model: "mnist_without_CNN"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 128)               100480    
_________________________________________________________________
dense_1 (Dense)              (None, 128)               16512     
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1290      
=================================================================
Total params: 118,282
Trainable params: 118,282
Non-trainable params: 0
```

##### CNN model achitecture

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0         
_________________________________________________________________
dropout (Dropout)            (None, 1600)              0         
_________________________________________________________________
dense (Dense)                (None, 10)                16010     
=================================================================
Total params: 34,826
Trainable params: 34,826
Non-trainable params: 0

```
#### model evaluation

this stage we evaluated all the model present in RDmodelstore using testdata obtained in Etl state, F1 score is obtained for all models and compared and evaluated. Model with great f1 score will be selected and uploaded to production modelstore

minimum f1 score for model evaluation is setting **min_f1_score** env which takes floting values


### model inferancing pipeline
-----------

There are 4 steps in the model building or training phase



#### production image sensor picture
In this stage we are simulating the data from image sensor and store the image in png

#### ETL

In this stage we will get image(png) from previous step of image sensor, the range of pixel value is 0 to 255. Normalization of dataset is applied for making the pixel value to range between 0 to 1 and ndarry is stored for prediction

#### COS

COS(cooridation service) is used to send the data for prediction and obtain the result from inferancing model

### inferancing 

this stage will take model from prod_model_store and expose api for prediction, takes ndarray of image as input and respond with prediction 

### raw data collection

this stage we store the real time image (ndarray) for furture Continues training. The result will be without label. **manual operation** Domain expert need to label the image and package it and upload to RD data source




execution steps for training and inferance of model
-----------

follow the below steps to simulate pipeline for dev env and production env
1. start the lister
  
  ```python3 start_listener.sh```
  
2. start the trainig process
  
  ```python3 train_execution.sh```
  
3. for inferancing, start the production process
  
  ```python3 production_execution.sh```


  

Continues training
-----------
Using the production data we can start the Continues-traing / trasfer-learning by manually labelling the data and uploading the data to datastore of dev env

1. for copying of production data and labeling is manual operation, assuming the labeled data is ready and ready for continues training

  ```python3 production_data_copy.sh``` 

2. for starting continues training

  ```python3 CT.sh```


proceduce for repeating the flow of execution : 
-----------
the process of training and inferancing of Model continues repeatatively 

follow step 2 and 3 in **execution steps for training and inferance of model** and **then Continues training**



Docker build
-----------

for building the docker , clone the repo and run the below cmd

#### docker build

```docker build --tag mnistpiple:v1 .```

#### docker run

```docker run -it mnistpiple:v1 bash```

#### execution

```cd /bd_build/```

follow the steps mentioned at **execution steps for training and inferance of model**



**Note**: testing was done on linus discribution of fedora and centos  
  

