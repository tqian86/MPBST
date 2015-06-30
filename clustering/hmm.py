#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import sys, os.path
pkg_dir = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(pkg_dir)

from MPBST import *

import numpy as np
from scipy.stats import multivariate_normal

class HMMSampler(BaseSampler):

    def __init__(self, num_states, record_best = True, cl_mode = False, cl_device = None, niter = 1000):
        """Initialize the base HMM sampler.
        """
        BaseSampler.__init__(self, record_best, cl_mode, cl_device, niter)

        self.data = None
        self.num_states = num_states
        self.uniq_states = np.linspace(1, self.num_states, self.num_states).astype(np.int64)
        self.trans_p_matrix = np.empty((num_states, num_states))
        self.trans_p_matrix.fill(1 / num_states)
        
    def read_csv(self, filepath, obsvar_names = ['obs'], header = True):
        """Read data from a csv file and check for observations.
        """
        self.data = pd.read_csv(filepath, compression = 'gzip')
        self.obs = self.data[obsvar_names]
        self.N = self.data.shape[0]
        self.states = np.random.randint(low = 1, high = self.num_states + 1, size = self.N)

class GaussianHMMSampler(HMMSampler):

    def __init__(self, num_states, record_best = True, cl_mode = False, cl_device = None, niter = 1000):
        """Initialize the base HMM sampler.
        """
        HMMSampler.__init__(self, num_states, record_best, cl_mode, cl_device, niter)
        
    def read_csv(self, filepath, obsvar_names = ['obs'], header = True):
        """Read data from a csv file and check for observations.
        """
        HMMSampler.read_csv(self, filepath, obsvar_names, header)
        self.dim = len(obsvar_names)
        self.means = np.zeros((self.num_states, self.dim)) # the mean vector of each state
        self.covs = np.array([np.eye(self.dim) for _ in xrange(self.num_states)]) # the covariance matrix of each state
        
        self.gaussian_mu0 = np.zeros(self.dim)
        self.gaussian_k0 = 1

        self.wishart_T0 = np.eye(self.dim)
        self.wishart_v0 = 1
        
    def do_inference(self, output_file = None):
        """Perform inference on parameters.
        """
        for i in xrange(self.niter):
            self._infer_means_covs(output_file)
            self._infer_states(output_file)
            self._infer_transp(output_file)
            print('Transitional matrix:\n', self.trans_p_matrix)
            print('Means:\n', self.means)
            print('States:\n', self.states)
        return

    def _infer_states(self, output_file):
        """Infer the state of each observation without OpenCL.
        """
        for nth in xrange(self.N):
            # set up sampling grid
            state_logp_grid = np.empty(shape = self.num_states)
            
            # loop over states
            for state in self.uniq_states:
                # compute the transitional probability from the previous state
                if nth == 0:
                    trans_prev_logp = np.log(1 / self.num_states)
                else:
                    prev_state = self.states[nth - 1]
                    trans_prev_logp = np.log(self.trans_p_matrix[prev_state-1, state-1])

                # compute the transitional probability to the next state
                if nth == self.N - 1:
                    trans_next_logp = np.log(1)
                else:
                    next_state = self.states[nth + 1]
                    trans_next_logp = np.log(self.trans_p_matrix[state-1, next_state-1])
                    
                emit_logp = multivariate_normal.logpdf(self.obs.iloc[nth], mean = self.means[state-1], cov = self.covs[state-1])
                state_logp_grid[state - 1] = trans_prev_logp + trans_next_logp + emit_logp

            # resample state
            self.states[nth] = sample(a = self.uniq_states, p = lognormalize(state_logp_grid))
            
        return

    def _infer_means_covs(self, output_file):
        """Infer the means of each hidden state without OpenCL.
        """
        for state in self.uniq_states:
            # get observations currently assigned to this state
            cluster_obs = np.array(self.obs.iloc[np.where(self.states == state)])
            n = cluster_obs.shape[0]
            if n == 0: continue
            
            # compute sufficient statistics
            mu = cluster_obs.mean()
            obs_deviance = cluster_obs - mu
            mu0_deviance = np.reshape(mu - self.gaussian_mu0, (1, self.dim))
            cov_obs = np.dot(obs_deviance.T, obs_deviance)
            cov_mu0 = np.dot(mu0_deviance.T, mu0_deviance)

            v_n = self.wishart_v0 + n
            k_n = self.gaussian_k0 + n
            T_n = self.wishart_T0 + cov_obs + cov_mu0 * self.gaussian_k0 * n / k_n

            # new mu is sampled from a multivariate t with the following parameters
            df = v_n - self.dim + 1
            mu_n = (self.gaussian_k0 * self.gaussian_mu0 + n * mu) / k_n
            Sigma = T_n / (k_n * df)

            # resample the new mean vector
            new_mu = multivariate_t(mu = mu_n, Sigma = Sigma, df = df)
            self.means[state-1] = new_mu
            # resample the covariance matrix
            new_cov = np.linalg.inv(wishart(Sigma = Sigma, df = v_n))
            self.covs[state-1] = new_cov
            
        return

    def _infer_transp(self, output_file):
        """Infer the transitional probabilities betweenn states without OpenCL.
        """
        for state_from in self.uniq_states:
            count_p = np.empty(self.num_states)
            pairs = zip(self.states[:self.N-1], self.states[1:])
            count_from_state = (self.states[:self.N-1] == state_from).sum()
            for state_to in self.uniq_states:
                count_p[state_to - 1] = pairs.count((state_from, state_to)) / count_from_state
                
            self.trans_p_matrix[state_from-1] = count_p
        return

hs = GaussianHMMSampler(num_states = 2)
hs.read_csv('./toydata/speed.csv.gz', obsvar_names = ['rt'])
hs.do_inference()
