#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:52:34 2020

@author: Jonathan.Venezia@ibm.com
"""

import unittest
from model import *

class ModelTest(unittest.TestCase):
    def test_01_train(self):
        saved_model = 'models/test-united_kingdom-0_1.joblib'
        ## train the model
        model_train(data_dir='data/cs-train',test=True)
        self.assertTrue(os.path.exists(saved_model))

    def test_02_load(self):
        all_data, all_models = model_load(training=False)

        country_list = ['all', 'eire', 'france', 'germany', 'hong_kong', 'netherlands',
                        'norway', 'portugal', 'singapore', 'spain', 'united_kingdom']
        self.assertEqual(list(all_data.keys()),country_list)
        self.assertEqual(list(all_models.keys()),country_list)


    def test_03_predict(self):
        country='all'
        year='2018'
        month='01'
        day='05'

        result = model_predict(country,year,month,day,all_models=None,test=True)

        y_pred = result['y_pred']
        self.assertEqual(len(y_pred),1)
        
if __name__ == '__main__':
    unittest.main()