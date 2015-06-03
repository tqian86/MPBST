#!/usr/bin/env python3
#!-*- coding: utf-8 -*-

from __future__ import print_function, division

import sys, os.path
pkg_dir = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(pkg_dir)

from MPBST import RegressionSampler

class LMMSampler(RegressionSampler):

    def do_inference(self, output_file=None):
        RegressionSampler.do_inference(self, output_file)


lmm = LMMSampler(record_best = True, cl_mode = False)
lmm.read_csv('./data/10group-100n.csv.gz')
lmm.do_inference()
print(lmm.obs)
