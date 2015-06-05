#!/usr/bin/env python
#!-*- coding: utf-8 -*-

from __future__ import print_function, division

import sys, os.path
pkg_dir = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(pkg_dir)

from MPBST import RegressionSampler

class LMMSampler(RegressionSampler):

    def __init__(self, record_best = True, cl_mode == False, cl_device = None):
        """Initialize the class.
        """
        RegressionSampler.__init__(self, record_best, cl_mode, cl_device)
       
    def do_inference(self, output_file=None):
        


    
lmm = LMMSampler(record_best = True, cl_mode = False)
lmm.read_csv('./data/10group-100n.csv.gz')
print(lmm.obs)
