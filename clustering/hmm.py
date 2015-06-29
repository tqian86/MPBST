#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import sys, os.path
pkg_dir = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(pkg_dir)

from MPBST import *

import numpy as np

class HMMSampler(BaseSampler):

    def __init__(self, num_states, record_best = True, cl_mode = False, cl_device = None, niter = 1000):
        """Initialize the base HMM sampler.
        """
        BaseSampler.__init__(self, record_best, cl_mode, cl_device, niter)

        self.data = None
        self.num_states = num_states
        self.trans_p_matrix = np.random.random((num_states, num_states))
        
    def read_csv(self, filepath, obsvar_names = ['obs'], header = True):
        """Read data from a csv file and check for observations.
        """
        self.data = pd.read_csv(filepath)
        self.obs = self.data[obsvar_names]
        self.N = self.data.shape[0]
        self.states = np.random.randint(low = 1, high = self.num_states + 1, size = self.N)

class GaussianHMMSampler(HMMSampler):

    def __init__(self, num_states, record_best = True, cl_mode = False, cl_device = None, niter = 1000):
        """Initialize the base HMM sampler.
        """
        HMMSampler.__init__(self, record_best, cl_mode, cl_device, niter)
        
    def read_csv(self, filepath, obsvar_names = ['obs'], header = True):
        """Read data from a csv file and check for observations.
        """
        HMMSampler.read_csv(self, filepath, obsvar_names, header)
        self.dim = len(obsvar_names)
        self.means = np.empty((self.num_states, self.dim)) # the mean vector of each state
        self.covs = np.empty((self.num_states, self.dim, self.dim)) # the covariance matrix of each state
        
    def do_inference(self, output_file = None):
        """Perform inference on parameters.
        """
        pass

    def _infer_states(self, output_file):
        """Infer the state of each observation without OpenCL.
        """
        return

    def _infer_means(self, output_file):
        """Infer the means of each hidden state without OpenCL.
        """
        return

    def _infer_covs(self, output_file):
        """Infer the covariance matrices of each hidden state without OpenCL.
        """
        return
        
        
hs = GaussianHMMSampler(num_states = 2)
hs.read_csv('./toydata/speed.csv.gz', obsvar_names = ['rt'])
