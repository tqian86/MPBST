#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import sys, os.path
pkg_dir = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(pkg_dir)

from MPBST.base.sampler import *

import itertools
import numpy as np
from scipy.stats import multivariate_normal
from collections import Counter

class HMMSampler(BaseSampler):

    def __init__(self, num_states, record_best = True, cl_mode = False, cl_device = None,
                 niter = 1000, thining = 0,
                 annealing = False, debug_mumble = False):
        """Initialize the base HMM sampler.
        """
        BaseSampler.__init__(self, record_best, cl_mode, cl_device, niter, thining,
                             annealing = annealing, debug_mumble = debug_mumble)

        self.data = None
        self.num_states = num_states
        self.uniq_states = np.linspace(1, self.num_states, self.num_states).astype(np.int64)
        self.trans_p_matrix = np.random.random((num_states, num_states))
        self.trans_p_matrix = self.trans_p_matrix / self.trans_p_matrix.sum(axis=1)
        
    def read_csv(self, filepath, obsvar_names = ['obs'], header = True):
        """Read data from a csv file and check for observations.
        """
        self.source_filepath = filepath
        self.source_dirname = os.path.dirname(filepath) + '/'
        self.source_filename = os.path.basename(filepath).split('.')[0]
        
        self.data = pd.read_csv(filepath, compression = 'gzip')
        self.obs = self.data[obsvar_names]
        self.N = self.data.shape[0]
        self.states = np.random.randint(low = 1, high = self.num_states + 1, size = self.N).astype(np.int32)

        if self.cl_mode:
            self.d_obs = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.array(self.obs, dtype=np.float32, order='C'))
            self.d_states = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                                      hostbuf = self.states.astype(np.int32))
        
class GaussianHMMSampler(HMMSampler):

    def __init__(self, num_states, record_best = True, cl_mode = False, cl_device = None, niter = 1000, thining = 0,
                 annealing = False, debug_mumble = False):
        """Initialize the base HMM sampler.
        """
        HMMSampler.__init__(self, num_states, record_best, cl_mode, cl_device, niter, thining, annealing, debug_mumble)

        if cl_mode:
            global cl
            import pyopencl as cl
            import pyopencl.array
            program_str = open(pkg_dir + 'MPBST/clustering/kernels/gaussian_hmm_cl.c', 'r').read()
            self.cl_prg = cl.Program(self.ctx, program_str).build()
            self.d_X, self.d_outcome_obs = None, None
        
    def read_csv(self, filepath, obsvar_names = ['obs'], header = True):
        """Read data from a csv file and check for observations.
        """
        HMMSampler.read_csv(self, filepath, obsvar_names, header)
        self.dim = len(obsvar_names)
        self.means = np.zeros((self.num_states, self.dim)) # the mean vector of each state
        self.covs = np.array([np.eye(self.dim) for _ in xrange(self.num_states)]) # the covariance matrix of each state
        
        self.gaussian_mu0 = np.zeros((1, self.dim))
        self.gaussian_k0 = 1

        self.wishart_T0 = np.eye(self.dim)
        self.wishart_v0 = 1

    def do_inference(self, output_folder = None):
        """Perform inference on parameters.
        """
        if output_folder is None:
            output_folder = self.source_dirname
        else:
            output_folder += '/'

        self.sample_fp = gzip.open(output_folder + '{0}-gaussian-hmm-state-means-covs.csv.gz'.format(self.source_filename), 'w')

        begin_time = time()
        for i in xrange(1, self.niter+1):
            self.set_temperature(iteration = i)
            if self.cl_mode:
                new_means, new_covs = self._infer_means_covs()
                new_trans_p = self._infer_trans_p()
                new_states = self._infer_states()                
            else:
                new_means, new_covs = self._infer_means_covs()
                new_trans_p = self._infer_trans_p()
                new_states = self._infer_states()

            if self.record_best:
                if self.auto_save_sample((new_means, new_covs, new_trans_p, new_states)):
                    print('Means: ', self.means, file=sys.stderr)
                    self.loglik = self.best_sample[1]
                    self._save_sample(iteration = i)
                if self.no_improvement(): break
                self.means, self.covs, self.trans_p_matrix, self.states = new_means, new_covs, new_trans_p, new_states
            else:
                self.means, self.covs, self.trans_p_matrix, self.states = new_means, new_covs, new_trans_p, new_states
                self.loglik = self._logprob((new_means, new_covs, new_trans_p, new_states))
                self._save_sample(iteration = i)
                
        self.total_time += time() - begin_time
        
        return self.gpu_time, self.total_time

    def _infer_states(self):
        """Infer the state of each observation without OpenCL.
        """
        # copy the states first, this is needed for record_best mode
        new_states = np.empty_like(self.states); new_states[:] = self.states[:]
        
        # emission probabilities can be calculated in one pass
        state_logp = np.empty((self.N, self.num_states))
        emit_logp = np.empty((self.N, self.num_states))
        for state in self.uniq_states:
            emit_logp[:,state-1] = multivariate_normal.logpdf(self.obs, mean = self.means[state-1], cov = self.covs[state-1])

        trans_logp = np.log(self.trans_p_matrix)
        # state probabilities need to be interated over
        for nth in xrange(self.N):

            if nth == 0:
                trans_prev_logp = [np.log(1 / self.num_states)] * self.num_states
            else:
                trans_prev_logp = trans_logp[new_states[nth - 1] - 1]

            if nth == self.N - 1:
                trans_next_logp = [np.log(1)] * self.num_states
            else:
                trans_next_logp = trans_logp[:, new_states[nth + 1] - 1]

            state_logp[nth] = trans_prev_logp + trans_next_logp

            #print(self.annealing_temp); raw_input()
            # resample state
            new_states[nth] = sample(a = self.uniq_states,
                                     p = lognormalize(x = state_logp[nth] + emit_logp[nth], temp = self.annealing_temp))
            
        return new_states

    def _cl_infer_states(self):
        """Infer the state of each observation without OpenCL.
        """
        # copy the states first, this is needed for record_best mode
        new_states = np.empty_like(self.states); new_states[:] = self.states[:]
        
        # emission probabilities can be calculated in one pass
        state_logp = np.empty((self.N, self.num_states), dtype=np.float32)
        emit_logp = np.empty((self.N, self.num_states), dtype=np.float32)

        d_means = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = self.means.astype(np.float32))
        d_cov_dets = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = np.linalg.det(self.covs).astype(np.float32))
        d_cov_invs = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = np.linalg.inv(self.covs).astype(np.float32))
        d_emit_logp = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = emit_logp.astype(np.float32))

        self.cl_prg.calc_emit_logp(self.queue, emit_logp.shape, None,
                                   self.d_obs, d_means, d_cov_dets, d_cov_invs, d_emit_logp,
                                   np.int32(self.dim))
        
        d_trans_p_matrix = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = self.trans_p_matrix.astype(np.float32))
        d_state_logp = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = state_logp)
        d_rand = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                           hostbuf = np.random.random(size = self.N).astype(np.float32))
        d_temp_logp = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                                hostbuf = np.empty(self.num_states, dtype=np.float32))

        cl.enqueue_copy(self.queue, emit_logp, d_emit_logp)

        trans_logp = np.log(self.trans_p_matrix)
        # state probabilities need to be interated over
        for nth in xrange(self.N):

            if nth == 0:
                trans_prev_logp = [np.log(1 / self.num_states)] * self.num_states
            else:
                trans_prev_logp = trans_logp[new_states[nth - 1] - 1]

            if nth == self.N - 1:
                trans_next_logp = [np.log(1)] * self.num_states
            else:
                trans_next_logp = trans_logp[:, new_states[nth + 1] - 1]

            state_logp[nth] = trans_prev_logp + trans_next_logp
                
            # resample state
            new_states[nth] = sample(a = self.uniq_states, p = lognormalize(state_logp[nth] + emit_logp[nth]))        
            
        return new_states

    def _infer_means_covs(self):
        """Infer the means of each hidden state without OpenCL.
        """
        obs = np.array(self.obs)
        new_means = np.empty_like(self.means)
        new_covs = np.empty_like(self.covs)
        
        for state in self.uniq_states:
            # get observations currently assigned to this state
            cluster_obs = obs[np.where(self.states == state)]
            n = cluster_obs.shape[0]

            # compute sufficient statistics
            if n == 0:
                mu = np.zeros(self.dim)
                cluster_obs = np.zeros((1, self.dim))
            else:
                mu = cluster_obs.mean(axis = 0)

            obs_deviance = cluster_obs - mu
            mu0_deviance = mu - self.gaussian_mu0
            cov_obs = np.dot(obs_deviance.T, obs_deviance)
            cov_mu0 = np.dot(mu0_deviance.T, mu0_deviance)

            v_n = self.wishart_v0 + n
            k_n = self.gaussian_k0 + n
            T_n = self.wishart_T0 + cov_obs + cov_mu0 * self.gaussian_k0 * n / k_n

            # new mu is sampled from a multivariate t with the following parameters
            df = v_n - self.dim + 1
            mu_n = (self.gaussian_k0 * self.gaussian_mu0 + n * mu) / k_n
            Sigma = T_n / (k_n * df)

            # means and covs are sampled independently since close solutions exist and
            # this reduces autocorrelation between samples
            
            # resample the new mean vector
            new_means[state-1] = multivariate_t(mu = mu_n, Sigma = Sigma, df = df)
            # resample the covariance matrix
            new_covs[state-1] = np.linalg.inv(wishart(Sigma = np.linalg.inv(T_n), df = v_n))

        # a hacky but surprisingly effective way to alleviate label switching
        reindex = new_means[:,0].argsort()
        new_means = new_means[reindex]
        new_covs = new_covs[reindex]
        return new_means, new_covs

    def _infer_trans_p(self):
        """Infer the transitional probabilities betweenn states without OpenCL.
        """
        # copy the current transition prob matrix; this is needed for record_best mode
        new_trans_p_matrix = np.empty_like(self.trans_p_matrix)
        new_trans_p_matrix[:] = self.trans_p_matrix[:]
        
        # make bigram pairs for easier counting
        pairs = zip(self.states[:self.N-1], self.states[1:])
        pair_count = Counter(pairs)
        
        for state_from in self.uniq_states:
            count_from_state = (self.states[:self.N-1] == state_from).sum()
            for state_to in self.uniq_states:
                new_trans_p_matrix[state_from - 1, state_to - 1] = (pair_count[(state_from, state_to)] + 1) / (count_from_state + self.num_states)
                
        return new_trans_p_matrix

    def _logprob(self, sample):
        """Calculate the log probability of model and data.
        """
        obs = np.array(self.obs)
        means, covs, trans_p, states = sample

        gpu_begin_time = time()

        if self.cl_mode:
            joint_logp = np.empty(self.N, dtype=np.float32)
            d_means = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = means.astype(np.float32))
            d_states = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = states.astype(np.int32))
            d_trans_p = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = trans_p.astype(np.float32))
            d_cov_dets = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = np.linalg.det(covs).astype(np.float32))
            d_cov_invs = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = np.linalg.inv(covs).astype(np.float32))
            d_joint_logp = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = joint_logp)

            self.cl_prg.calc_joint_logp(self.queue, joint_logp.shape, None,
                                        self.d_obs, d_states, d_trans_p, d_means, d_cov_dets, d_cov_invs, d_joint_logp,
                                        np.int32(self.num_states), np.int32(self.dim))
            cl.enqueue_copy(self.queue, joint_logp, d_joint_logp)

        else:
            joint_logp = np.empty(self.N)
            # calculate transition probabilities first
            joint_logp[0] = np.log(1 / self.num_states)
            joint_logp[1:] = np.log(trans_p[states[:self.N-1] - 1, states[1:] - 1])
            # emission probs
            joint_logp = joint_logp + np.array([multivariate_normal.logpdf(obs[i], mean = means[states[i]-1], cov = covs[states[i]-1]) for i in xrange(self.N)])
            
        self.gpu_time += time() - gpu_begin_time
            
        return joint_logp.sum()

    def _save_sample(self, iteration):
        """Save the means and covariance matrices from the current iteration.
        """
        if not self.header_written: 
            header = ['iteration', 'loglik', 'dimension', 'state'] + ['mu_{0:d}'.format(_) for _ in range(1, self.dim+1)]
            header += ['cov_{0:d}_{1:d}'.format(*_) for _ in itertools.product(*[range(1, self.dim+1)] * 2)]
            header += ['trans_p_to_{0:d}'.format(_) for _ in self.uniq_states]
            header += ['has_obs_{0:d}'.format(_) for _ in range(1, self.N + 1)]
            print(*header, sep = ',', file = self.sample_fp)
            self.header_written = True

        for state in self.uniq_states:
            row = [iteration, self.loglik, self.dim, state] + list(self.means[state-1]) + list(np.ravel(self.covs[state-1])) + list(np.ravel(self.trans_p_matrix[state-1]))
            row += list((self.states == state).astype(np.bool).astype(np.int0))
            print(*row, sep = ',', file = self.sample_fp)
                
        return
    

if __name__ == '__main__':
    hs = GaussianHMMSampler(num_states = 2, niter = 2000, record_best = False, cl_mode=True, debug_mumble = True)
    hs.read_csv('./toydata/speed.csv.gz', obsvar_names = ['rt'])
    gt, tt = hs.do_inference()
    print(gt, tt)
