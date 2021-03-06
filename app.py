#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 00:47:13 2020

@author: Jonathan.Venezia@ibm.com
"""

import argparse
from flask import Flask, jsonify, request
from flask import render_template, send_from_directory
import os
import re
import numpy as np

from model import model_train, model_predict
with open('__version__','r+') as f:
    MODEL_VERSION = f.read()
    f.close

app = Flask(__name__)

@app.route("/")
def hello():
    html = "<h3>Hello {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>"
    return html.format(name=os.getenv("NAME", "world"), hostname=socket.gethostname())

@app.route('/predict', methods=['GET','POST'])
def predict():
    if not request.json:
        print("ERROR: API (predict): did not receive request data")
        return jsonify([])

    if 'query' not in request.json:
        print("ERROR API (predict): received request, but no 'query' found within")
        return jsonify([])
    
    query = request.json['query']
    
    _result = model_predict(*query)

    result = {}

    for key,item in _result.items():
        if isinstance(item,np.ndarray):
            result[key] = item.tolist()
        else:
            result[key] = item

    return(jsonify(result["y_pred"]))

@app.route('/train', methods=['GET','POST'])
def train():
    if not request.json:
        print("ERROR: API (train): did not receive request data")
        return jsonify(False)

    ## set the test flag
    test = False
    if 'mode' in request.json and request.json['mode'] == 'test':
        test = True
    query = request.json['query']
    print("... training model")
    model_train(data_dir=query,test=test)
    print("... training complete")

    return(jsonify(True))

@app.route('/logs/<filename>',methods=['GET'])
def logs(filename):
    if not re.search(".log",filename):
        print("ERROR: API (log): file requested was not a log file: {}".format(filename))
        return jsonify([])

    log_dir = os.path.join(".","logs")
    if not os.path.isdir(log_dir):
        print("ERROR: API (log): cannot find log dir")
        return jsonify([])

    file_path = os.path.join(log_dir,filename)
    if not os.path.exists(file_path):
        print("ERROR: API (log): file requested could not be found: {}".format(filename))
        return jsonify([])

    return send_from_directory(log_dir, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=8080)