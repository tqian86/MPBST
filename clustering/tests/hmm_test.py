#!/usr/bin/env python

from __future__ import print_function, division
from MPBST.clustering.hmm import *


if __name__ == '__main__':
    hs = GaussianHMMSampler(num_states = 10, niter = 200, record_best = False, cl_mode=False, debug_mumble = True)
    hs.read_csv('../toydata/speed.csv.gz', obs_vars = ['rt'], group = 'corr')
    gt, tt = hs.do_inference()
    print(gt, tt)
