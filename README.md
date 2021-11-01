# mnist_pipeline
mlops pipeline for mnist data

Pipeline overview: 
![alt text](https://raw.githubusercontent.com/poornachandratejasvi/mnist_pipeline/main/picture/mnist-overall.png "pipeline overviw")

Pipeline execution flow: 
![alt text](https://raw.githubusercontent.com/poornachandratejasvi/mnist_pipeline/main/picture/mnist-detailed.png "execution flow")


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


  

