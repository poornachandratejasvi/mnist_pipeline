

import os
from flask import Flask, request, redirect, url_for
# from werkzeug import secure_filename
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
import tarfile
from PIL import Image
from datetime import datetime

global modelPath
model_store = os.environ.get("model_store_path", "/tmp/production/prod_modelstore/")
real_data= os.environ.get("model_store_path", "/tmp/production/prod_realtimedata/")
UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = set(['py','.gz','tar.gz','tar','npz','png'])

if not os.path.exists(real_data):
    os.makedirs(real_data)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def modelExtract(file_location):
    if not os.path.exists(file_location):
        os.makedirs(file_location)
    file = tarfile.open(file_location)
    file.extractall(model_store)
    file.close()
    
def modelprediction(file_location):
    model = tf.keras.models.load_model(model_store+"/model/")
    print(model.summary())
    with np.load(file_location) as data:
        x_test = data['imgs']
        y_test = data['labels']
    x_test = np.expand_dims(x_test, -1)
    predictions = model.predict(x_test)
    return predictions

@app.route("/uploadmodel", methods=['PUT', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['files']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_location=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_location)
            modelExtract(file_location)
            modelPath=file_location
            return "successfully uploaded the model to modelstore"
        else:
            return "not valid file formate",201
    return "error",201

@app.route("/imagepredict", methods=['PUT', 'POST'])
def imagepredict():
    if request.method == 'POST':
        # file = request.files['files']
        # if file and allowed_file(file.filename):
        #     filename = secure_filename(file.filename)
        #     file_location=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #     file.save(file_location)
        #     pred=modelprediction(file_location)
            # print(np.argmax(pred[0]))
            # return pred
        img = Image.open(request.files['files'].stream).convert("L")
        img = img.resize((28,28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,28,28,1)
        im2arr = tf.keras.utils.normalize(im2arr, axis=1)
        # im2arr = 255-im2arr
        # im2arr /= 255
        model = tf.keras.models.load_model(model_store+"/model/")
        # with graph.as_default():
        y_pred = model.predict_classes(im2arr)
        # else:
        #     return "not valid file formate",201
        return 'Predicted Number: ' + str(y_pred[0])
    return "error",201

@app.route("/predict", methods=['PUT', 'POST'])
def perdict():
    if request.method == 'POST':
        # file = request.files['files']
        # if file and allowed_file(file.filename):
        #     filename = secure_filename(file.filename)
        #     file_location=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #     file.save(file_location)
        #     pred=modelprediction(file_location)
            # print(np.argmax(pred[0]))
            # return pred
        data = request.json
        x_test = np.array(data)
        # img = Image.open(request.files['files'].stream).convert("L")
        # img = img.resize((28,28))
        # im2arr = np.array(img)
        # im2arr = im2arr.reshape(1,28,28,1)
        # im2arr = tf.keras.utils.normalize(im2arr, axis=1)
        # im2arr = 255-im2arr
        # im2arr /= 255
        model = tf.keras.models.load_model(model_store+"/model/")
        # with graph.as_default():
        y_pred = model.predict_classes(x_test)
        cur_time=datetime.now().strftime("%Y%m%d_%H%M%S%f")
        if not os.path.exists(real_data):
            os.makedirs(real_data)
        np.savez_compressed(real_data+"/real_data"+cur_time+".npz", imgs=x_test)
        # else:
        #     return "not valid file formate",201
        return 'Predicted Number: ' + str(y_pred[0])
    return "error",201

if __name__ == "__main__":
    app.run(host='localhost', port=8082,debug = True)
    
