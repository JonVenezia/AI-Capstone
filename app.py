#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 00:47:13 2020

@author: Jonathan.Venezia@ibm.com
"""

from flask import Flask, jsonify, request, send_from_directory
import joblib
import socket
import json
import pandas as pd
import os
import re
from model import model_train,  model_predict

app = Flask(__name__)

@app.route("/")
def hello():
    html = "<h3>Hello {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>"
    return html.format(name=os.getenv("NAME", "world"), hostname=socket.gethostname())

@app.route('/predict', methods=['GET','POST'])
def predict():
    
    ## input checking
    if not request.json:
        print("ERROR: API (predict): did not receive request data")
        return jsonify([])

    query = request.json
    query = pd.DataFrame(query)
    
    if len(query.shape) == 1:
         query = query.reshape(1, -1)

    y_pred = model.predict(query)
    
    return(jsonify(y_pred.tolist()))

def train():
    """
    basic predict function for the API
    the 'mode' flag provides the ability to toggle between a test version and a
    production verion of training
    """

    ## check for request data
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
    """
    API endpoint to get logs
    """

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
    saved_model = 'aavail-rf.joblib'
    model = joblib.load(saved_model)
    app.run(host='0.0.0.0', port=8080,debug=True)