#!/bin/bash
python3 /bd_build/py_pipe_production/1_sensorData/SensorImage.py 
python3 /bd_build/py_pipe_production/2_ETL/etl.py
python3 /bd_build/py_pipe_production/3_cos/cos.py
