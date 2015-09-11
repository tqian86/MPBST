#!/usr/bin/env python
#-*- coding: utf-8 -*-
from __future__ import print_function, division

import argparse
import sys

from MPBST.clustering.hmm import *

parser = argparse.ArgumentParser()
parser.add_argument("data", type = str, help = "the data source file, expected in csv format and gzipped")
parser.add_argument("--outcome", nargs='+', action = 'append', required = True,
                    help = "specify an outcome variable that is independent of other outcome variables; more than one variable name can follow a single --outcome, separated by space \
                    which will be treated as a joint multivariate outcome variable set. More than one --outcome can be specified, which will be treated as iid.")
parser.add_argument("--group", type=str, help = "optinally specify the grouping variable where each unique value refers to a group with its unique transition matrix to be estimated")
parser.add_argument("-k", type = int, required = True, help = "the number of states")
parser.add_argument("-n", type = int, default = 1000, help = "the number of iterations (maximum iterations when performing stochastic search)")
parser.add_argument("--search", action="store_true", help = "use stochastic search instead of Gibbs sampling")
parser.add_argument("--opencl", action="store_true", help = "use opencl optimization")
parser.add_argument("--debug", "-d", action="store_true", help = "turn on debug mode")
parser.add_argument("-t", type = int, default = 250, help = "in stochastic search model, the number of iterations during which the loglikelihood of data (or joint with model) may not improve")
args = parser.parse_args()

print(args.__dict__)

if __name__ == '__main__':
    hs = GaussianHMMSampler(num_states = args.k, sample_size = args.n, search = args.search, cl_mode = args.opencl, debug_mumble = args.debug, search_tolerance = args.t)
    hs.read_csv(args.data, obs_vars = args.outcome, group = args.group)
    gt, tt = hs.do_inference()
    print(gt, tt)
