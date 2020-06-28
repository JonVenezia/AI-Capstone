#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:13:39 2020

@author: Jonathan.Venezia@ibm.com
"""

import unittest
import getopt
import sys
import os

try:
    optlist, args = getopt.getopt(sys.argv[1:],'v')
except getopt.GetoptError:
    print(getopt.GetoptError)
    print(sys.argv[0] + "-v")
    print("... the verbose flag (-v) may be used")
    sys.exit()

VERBOSE = True
RUNALL = False

sys.path.append(os.path.realpath(os.path.dirname(__file__)))

for o, a in optlist:
    if o == '-v':
        VERBOSE = True

# api tests
from ApiTests import *
ApiTestSuite = unittest.TestLoader().loadTestsFromTestCase(ApiTest)

## model tests
from ModelTests import *
ModelTestSuite = unittest.TestLoader().loadTestsFromTestCase(ModelTest)

## logger tests
from LogTests import *
LogTestSuite = unittest.TestLoader().loadTestsFromTestCase(LogTests)

MainSuite = unittest.TestSuite([LogTestSuite,ModelTestSuite,ApiTestSuite])