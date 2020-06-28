#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 23:15:32 2020

@author: Jonathan.Venezia@ibm.com
"""

import time,os,csv,uuid
from datetime import date

if not os.path.exists(os.path.join(".","logs")):
    os.mkdir("logs")

def update_train_log(tag,dt_range,eval_test,runtime,MODEL_VERSION,MODEL_VERSION_NOTE,test=False):
    
    today = date.today()
    if test:
        logfile = os.path.join("logs",f"{tag}-train-test.log")
    else:
        logfile = os.path.join("logs",f"sl-{tag}-train-{today.year}-{today.month}.log")


    header = ['unique_id','timestamp','dt_range','eval_test','model_version',
              'model_note','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),int(time.time()),dt_range,eval_test,MODEL_VERSION,
                            MODEL_VERSION_NOTE,runtime])
        writer.writerow(to_write)

def update_predict_log(country,y_pred,y_proba,target_date,runtime, MODEL_VERSION, test=False):

    today = date.today()
    if test:
        logfile = os.path.join("logs","predict-test.log")
    else:
        logfile = os.path.join("logs",f"predict-{today.year}-{today.month}.log")

    header = ['unique_id','timestamp','y_pred','y_proba','country','target_date','model_version','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),int(time.time()),y_pred,y_proba,country,target_date,
                            MODEL_VERSION,runtime])
        writer.writerow(to_write)
