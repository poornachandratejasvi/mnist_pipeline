#!/bin/bash
export model_append="True"
python3 /bd_build/py_pipe_dev/1_download/download.py
python3 /bd_build/py_pipe_dev/2_ETL/etl.py
python3 /bd_build/py_pipe_dev/3_model/mnist_simple.py
python3 /bd_build/py_pipe_dev/3_model/mnist_cnn.py
python3 /bd_build/py_pipe_dev/4_evaluation/evaluation.py
