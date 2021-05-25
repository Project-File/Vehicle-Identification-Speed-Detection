from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import glob
import os
import cv2
import numpy as np
import pandas as pd
import detect_object
from shutil import copyfile
import shutil
from distutils.dir_util import copy_tree

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

for f in os.listdir("static\\similar_images\\"):
    os.remove("static\\similar_images\\"+f)

print('Model loaded. Check http://127.0.0.1:5000/')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        get_detected_object = detect_object(file_path)
        return get_detected_object
    return None


if __name__ == '__main__':
    app.run(debug=True)
