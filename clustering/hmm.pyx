#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Gibbs samplers of Hidden Markov Models with OpenCL support.

This module implements Gibbs samplers of Hidden Markov Models (HMMs). An HMM is used to model the 
sequence of hidden states that underlie a sequence of observed random outcomes. Each hidden state 
is modeled as a probabilistic distribution that "generates" observed outcomes. Depending on the 
specific HMM that a modeler chooses to use (e.g., GaussianHMMSampler), this probabilistic distribution 
can be gaussian/normal, categorical, or of other forms. In addition, transition probabilities between
hidden states are modeled as categorical distributions with beta priors, regardless of which HMM 
sampler is used.

As is standard with all tools in MPBST, this module supports at least both vanilla Gibbs sampling
and stochastic search.

OpenCL support
---------
The current set of HMMs come with (limited yet significantly helpful) OpenCL support. Due to the
sequential dependencies between the states in an HMM, the overhead incurred by OpenCL commands in
inferring states makes it an unwise strategy in speeding up the overall sampling process. Instead,
OpenCL optimization is applied to the evaluation of the joint log probability of the inferred states
and the data, which tends to be the most costly step of the sampling process.

Usage
-----
Use one of the subclasses, e.g., GaussianHMMSampler, directly. The base class HMMSampler is intended for
development only. 

Initialize a sampler by constructing an HMM instance. For example,

hmm = GaussianHMMSampler(num_states = 4)

See docs for the list of optional arguments.

Input data
----------
Input data can be read using the standard "read_csv" method of an initialized sampler. For example:

hmm.read_csv("./data/data.csv", obs_vars = ["obs"], group = None, header = True, timestamp = None)

See the docstring of read_csv for details.


Available models:
----------------
- Grouped HMM with k-dimensional Gaussian/Normal emission probability distributions 
  (K can be anything from 1 to whatever makes sense in your data)

Planned models:
--------------
- HMM with categorical emission probability distributions
"""

from __future__ import print_function, division

import sys, os.path, gzip
import cython; cimport cython

import MPBST
from MPBST import *
import itertools
import numpy as np; cimport numpy as np
from scipy.stats import multivariate_normal
from collections import Counter
from time import time
from libc.math cimport exp, log, pow
from libc.stdlib cimport rand, RAND_MAX

def multivariate_t(np.ndarray mu, np.ndarray Sigma, int df, size=None):
    '''
    Output:
    Produce size samples of d-dimensional multivariate t distribution
    Input:
    mu = mean (d dimensional numpy array or scalar)
    Sigma = scale matrix (dxd numpy array)
    df = degrees of freedom
    size = # of samples to produce
    '''
    cdef int d = len(Sigma)
    Z = np.random.multivariate_normal(np.zeros(d), Sigma, size)
    g = np.repeat(np.random.gamma(df/2, 2/df, size), d).reshape(Z.shape)
    return mu + Z/np.sqrt(g)


def wishart(int df, np.ndarray Sigma, size=None):

    cdef int dim = len(Sigma)
    Z = np.random.multivariate_normal(np.zeros(dim), Sigma, df)
    return Z.T.dot(Z)

@cython.boundscheck(False)
cdef object sample(a, np.ndarray[np.float_t, ndim = 1] p):
    """Step sample from a discrete distribution using CDF
    """
    if (len(a) != len(p)):
        raise Exception('a != p')

    cdef double p_sum = 0
    cdef int i, p_length = len(p)

    for i in xrange(p_length):
        p_sum += p[i]

    for i in xrange(p_length):
        p[i] = p[i] /  p_sum
        
    cdef double r = rand() / RAND_MAX #random.random()
    cdef double total = 0           # range: [0,1]
    for i in xrange(p_length):
        total += p[i]
        if total > r:
            return a[i]
    return a[p_length - 1]

@cython.boundscheck(False)
cdef np.ndarray[np.float_t, ndim=1] lognormalize(np.ndarray[np.float_t, ndim=1] x, double temp = 1):
    """Normalize a vector of logprobabilities to probabilities that sum up to 1.
    Optionally accepts an annealing temperature that does simple annealing.
    """
    cdef int i, x_length = x.shape[0]
    cdef double x_max, x_sum = 0
    for i in xrange(x_length):
        if i == 0: x_max = x[i]
        elif x[i] > x_max: x_max = x[i]

    for i in xrange(x_length):
        x[i] = pow(exp(x[i] - x_max), temp)
        x_sum += x[i]

    for i in xrange(x_length):
        x[i] /= x_sum

    return x

class HMMSampler(BaseSampler):

    def __init__(self, int num_states, record_best = True, cl_mode = False, cl_device = None,
                 int sample_size = 1000, annealing = False, debug_mumble = False):
        """Initialize the base HMM sampler.
        """
        BaseSampler.__init__(self,
                             record_best = record_best,
                             cl_mode = cl_mode,
                             cl_device = cl_device,
                             sample_size = sample_size, 
                             annealing = annealing,
                             debug_mumble = debug_mumble)

        self.data = None
        self.num_states = num_states
        cdef np.ndarray[long, ndim=1] uniq_states = np.linspace(1, self.num_states, self.num_states).astype(np.int64)
        self.uniq_states = uniq_states
        
        cdef np.ndarray[np.float_t, ndim=2] trans_p_matrix
        trans_p_matrix = np.random.random((num_states+1, num_states+1))
        trans_p_matrix = trans_p_matrix / trans_p_matrix.sum(axis=1)
        self.trans_p_matrix = trans_p_matrix

        cdef int SEQ_BEGIN, SEQ_END
        SEQ_BEGIN, SEQ_END = 1, 2
        self.SEQ_BEGIN, self.SEQ_END = SEQ_BEGIN, SEQ_END
        
        self.str_output_lines = []
        
    def read_csv(self, str filepath, list obs_vars = ['obs'], str group = None, str timestamp = None, header = True):
        """Read data from a csv file and check for observations. 

        Time series data are expected to be stored in columns, rather than rows of various lengths.
        That is, if a sequence of random outcomes is 1,2,3,4,5, the csv should look like:
        
        obs
        1
        2
        3
        4
        5

        Rather than "1,2,3,4,5". The preferred format is more flexible in many situations, allowing
        the specification of multiple outcome variables, grouping factors, and timestamps.

        ----------

        - The specification of outcome variables follows this rule: top-level variables and 
        variable groups are assumed to be i.i.d. Second-level (inside a sublist) variables are 
        assumed to be multivariate normals. For example:

        obs_vars = ['obs1', 'obs2']

        implies that the emission probability distribution is a joint of two normals, while

        obs_vars = [['obs1', 'obs2']] 

        implies that a two-dimensional normal serves as the emission probability distribution.

        ----------

        - Groups can be specified via the "group" argument. The argument expects the name of the
        column whose values are labels of groups. The current HMMs only support one grouping
        factor. Future versions may introduce support for nested groups. 

        For instance, if "group" is specified as the grouping variable, and the data set

        outcome,group
        1,1
        2,1
        3,1
        3,2
        4,2
        5,2

        will be parsed as two observed sequences. Importantly, the first "3" of Group 2 will have
        a preceeding state known as "boundary marker", instead of "3".

        ----------

        - A timestamp variable can be specified using the "timestamp" argument. This provides
        the flexibility that allows the input data to be supplied to the sampler in random order,
        as long as the correct ordering information is in "timestamp". "timestamp" expects the
        name of the column that provides such ordering information.

        If "timestamp" is left at the default value None, the line order of the CSV is used
        as the timestamp order.
        
        """
        BaseSampler.read_csv(self,
                             filepath = filepath,
                             obs_vars = obs_vars,
                             header = header)

        # create timestamp using row number if no timestamp is supplied
        if timestamp is None:
            timestamp = '_time_stamp_'
            self.original_data['_time_stamp_'] = range(self.N)

        # define boundary mask
        cdef np.ndarray[np.int_t, ndim = 1] boundary_mask
        
        if group is None:
            self.original_data.sort(columns = timestamp, inplace=True)
            self.data = self.original_data[obs_vars]
            boundary_mask = np.zeros(self.N, dtype=np.int)
            boundary_mask[0] = self.SEQ_BEGIN
            boundary_mask[-1] = self.SEQ_END
            self.boundary_mask = boundary_mask
        else:
            self.original_data.sort(columns = [group, timestamp], inplace=True)
            self.data = self.original_data[obs_vars]
            boundary_mask = np.zeros(self.N, dtype=np.int)
            rle_values, rle_lengths = zip(*[(k, len(list(g))) for k, g in itertools.groupby(self.original_data[group])])
            boundary_begins = np.array(rle_lengths).cumsum() - rle_lengths
            boundary_ends = np.array(rle_lengths).cumsum() - 1
            boundary_mask[boundary_begins] = self.SEQ_BEGIN
            boundary_mask[boundary_ends] = self.SEQ_END
            self.boundary_mask = boundary_mask

        # define states
        cdef np.ndarray[np.int_t, ndim = 1] states
        states = np.random.randint(low = 1, high = self.num_states + 1, size = self.N).astype(np.int)
        self.states = states

        if self.cl_mode:
            self.d_obs = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.array(self.data, dtype=np.float32, order='C'))

    @cython.boundscheck(False)
    def _infer_trans_p(self):
        """Infer the transitional probabilities betweenn states without OpenCL.
        """
        # copy the current transition prob matrix; this is needed for record_best mode
        cdef np.ndarray[np.float_t, ndim = 2] new_trans_p_matrix = np.empty_like(self.trans_p_matrix)
        new_trans_p_matrix[:] = self.trans_p_matrix[:]
        
        # make bigram pairs for easier counting
        cdef list pairs = zip(self.states[:self.N-1], self.states[1:])

        # add also pairs made up by boundary marks 
        cdef np.ndarray[np.int_t, ndim = 1] begin_states, end_states
        cdef long SEQ_BEGIN = self.SEQ_BEGIN, SEQ_END = self.SEQ_END
        
        begin_states = self.states[self.boundary_mask == SEQ_BEGIN]
        pairs.extend(zip([0] * begin_states.shape[0], begin_states))
        end_states = self.states[self.boundary_mask == SEQ_END]
        pairs.extend(zip(end_states, [0] * begin_states.shape[0]))

        pair_count = Counter(pairs)

        cdef long state_from, state_to, count_from_state
        for state_from in np.insert(self.uniq_states, 0, 0):
            count_from_state = np.sum([_[0] == state_from for _ in pairs])
            for state_to in np.insert(self.uniq_states, 0, 0):
                new_trans_p_matrix[state_from, state_to] = (pair_count[(state_from, state_to)] + 1) / (count_from_state + self.num_states + 1)

        new_trans_p_matrix[0, 0] = 0
        new_trans_p_matrix[0] = new_trans_p_matrix[0] / new_trans_p_matrix[0].sum()
                
        return new_trans_p_matrix

cdef np.ndarray[np.int_t, ndim=2] _gaussian_mu0(int dim):
    return np.zeros((1, dim))

cdef int _gaussian_k0(int dim):
    return 1

cdef np.ndarray[np.int_t, ndim=2] _wishart_T0(int dim):
    return np.eye(dim)

cdef int _wishart_v0(int dim):
    return 1

class GaussianHMMSampler(HMMSampler):

    def __init__(self, int num_states, record_best = True, cl_mode = False, cl_device = None,
	         int sample_size = 1000, annealing = False, debug_mumble = False):
        """Initialize the base HMM sampler.
        """
        HMMSampler.__init__(self, num_states = num_states,
                            record_best = record_best,
                            cl_mode = cl_mode, cl_device = cl_device,
                            sample_size = sample_size,
                            annealing = annealing,
                            debug_mumble = debug_mumble)

        if cl_mode:
            global cl
            import pyopencl as cl
            import pyopencl.array
            program_str = open(MPBST.__path__[0] + '/clustering/kernels/gaussian_hmm_cl.c', 'r').read()
            self.cl_prg = cl.Program(self.ctx, program_str).build()
        
    def read_csv(self, str filepath, list obs_vars = ['obs'], str group = None, str timestamp = None, header = True):
        """Read data from a csv file and set up means and covariance matrices for the Gaussian
        generative model.
        """
        flat_obs_vars = list(np.hstack(obs_vars))
        self.obs_vars = obs_vars; self.flat_obs_vars = flat_obs_vars

        HMMSampler.read_csv(self, filepath, obs_vars = flat_obs_vars, group = group,
                            timestamp = timestamp, header = header)

        cdef long s

        self.num_var = len(flat_obs_vars)
        self.num_cov_var = np.sum([len(_) ** 2 if type(_) is list else 1 for _ in obs_vars])
        self.num_var_set = len(obs_vars)

        cdef list means, covs
        means = [[np.zeros(len(_)) if type(_) is list else np.zeros(1) for _ in obs_vars] for s in xrange(self.num_states)]
        covs = [[np.eye(len(_)) if type(_) is list else np.eye(1) for _ in obs_vars] for s in xrange(self.num_states)]
        self.means, self.covs = means, covs
        
        return True

    @cython.boundscheck(False)
    def _infer_states(self):
        """Infer the state of each observation without OpenCL.
        """
        cdef int num_states = self.num_states
        
        # copy the states first, this is needed for record_best mode
        cdef np.ndarray[np.int_t, ndim=1] new_states = np.empty(self.states.shape, dtype=np.int);
        new_states[:] = self.states[:]
        
        # set up grid for storing log probabilities
        cdef np.ndarray[np.float_t, ndim=2] emit_logp = np.empty((self.N, num_states))

        cdef long var_set_idx, state_idx, state, nth
        cdef long num_var_set = self.num_var_set
        cdef np.ndarray[np.int_t, ndim=1] uniq_states = self.uniq_states
        
        # emission probabilities can be calculated in one pass
        for var_set_idx in xrange(num_var_set):
            var_set = self.obs_vars[var_set_idx]
            obs_set = self.data[var_set]
            for state_idx in xrange(num_states):
                state = uniq_states[state_idx]
                if var_set_idx == 0: emit_logp[:, state-1] = 0
                emit_logp[:,state-1] += multivariate_normal.logpdf(obs_set,
                                                                   mean = self.means[state-1][var_set_idx],
                                                                   cov = self.covs[state-1][var_set_idx])
       
        # state probabilities need to be interated over
        cdef np.ndarray[np.float_t, ndim=2] trans_logp = np.log(self.trans_p_matrix)
        cdef np.ndarray[np.float_t, ndim=1] trans_prev_logp, trans_next_logp, state_logp
        cdef long is_beginning, is_end

        for nth in xrange(self.N):
            is_beginning = self.boundary_mask[nth] == self.SEQ_BEGIN
            is_end = self.boundary_mask[nth] == self.SEQ_END
            
            if is_beginning:
                trans_prev_logp = trans_logp[0, 1:]
            else:
                trans_prev_logp = trans_logp[new_states[nth - 1], 1:]

            if is_end:
                trans_next_logp = trans_logp[1:, 0]
            else:
                trans_next_logp = trans_logp[1:, new_states[nth + 1]]

            state_logp = trans_prev_logp + trans_next_logp

            gpu_time = time()
            # resample state
            new_states[nth] = sample(a = uniq_states,
                                     p = lognormalize(x = state_logp + emit_logp[nth], temp = self.annealing_temp))
            self.gpu_time += time() - gpu_time

        return new_states

    @cython.boundscheck(False)
    def _infer_means_covs(self):
        """Infer the means of each hidden state without OpenCL.
        """
        cdef long s, var_set_idx, obs_dim, state, state_idx, n, v_n, df, k_n
        cdef np.ndarray obs, cluster_obs
        cdef long num_states = self.num_states
        
        cdef list new_means = [[np.zeros(len(_)) if type(_) is list else np.zeros(1) for _ in self.obs_vars] for s in xrange(num_states)]
        cdef list new_covs = [[np.eye(len(_)) if type(_) is list else np.eye(1) for _ in self.obs_vars] for s in xrange(num_states)]

        for var_set_idx in xrange(self.num_var_set):
            var_set = self.obs_vars[var_set_idx]
            if type(var_set) is str: obs_dim = 1
            else: obs_dim = len(var_set)
            obs = np.array(self.data[var_set])

            for state_idx in xrange(num_states):
                state = self.uniq_states[state_idx]
                
                # get observations currently assigned to this state
                cluster_obs = obs[np.where(self.states == state)]
                n = cluster_obs.shape[0]
                # compute sufficient statistics
                if n == 0:
                    mu = np.zeros(obs_dim)
                    cluster_obs = np.zeros((1, obs_dim))
                else:
                    mu = cluster_obs.mean(axis = 0)

                obs_deviance = cluster_obs - mu
                mu0_deviance = mu - _gaussian_mu0(dim = obs_dim)
                cov_obs = obs_deviance.T.dot(obs_deviance)
                cov_mu0 = np.dot(mu0_deviance.T, mu0_deviance)

                v_n = _wishart_v0(obs_dim) + n
                k_n = _gaussian_k0(obs_dim) + n
                T_n = _wishart_T0(obs_dim) + cov_obs + cov_mu0 * _gaussian_k0(obs_dim) * n / k_n

                # new mu is sampled from a multivariate t with the following parameters
                df = v_n - obs_dim + 1
                mu_n = (_gaussian_k0(obs_dim) * _gaussian_mu0(obs_dim) + n * mu) / k_n
                Sigma = T_n / (k_n * df)

                # means and covs are sampled independently since close solutions exist and
                # this reduces autocorrelation between samples
            
                # resample the new mean vector
                new_means[state-1][var_set_idx][:] = multivariate_t(mu = mu_n, Sigma = Sigma, df = df)
                # resample the covariance matrix
                new_covs[state-1][var_set_idx][:] = np.linalg.inv(wishart(Sigma = np.linalg.inv(T_n), df = v_n))

        # a hacky but surprisingly effective way to alleviate label switching
        m = np.array([new_means[_][0][0] for _ in xrange(num_states)])
        reindex = m.argsort()
        new_means = [new_means[i] for i in reindex]
        new_covs = [new_covs[i] for i in reindex]
        return new_means, new_covs

    @cython.boundscheck(False)
    def _logprob(self, sample):
        """Calculate the log probability of model and data.
        """
        cdef np.ndarray[np.float_t, ndim = 2] trans_p
        cdef np.ndarray[np.float32_t, ndim = 1] joint_logp
        cdef np.ndarray[np.int_t, ndim = 1] states
        cdef list means, covs
        cdef long var_set_idx, i, state
        cdef float gpu_begin_time
        
        means, covs, trans_p, states = sample

        if self.cl_mode:
            var_set_dim = np.array([len(_) if type(_) is list else 1 for _ in self.obs_vars])
            var_set_linear_offset = np.hstack(([0], var_set_dim.cumsum()[:self.num_var_set - 1]))
            var_set_sq_offset = np.hstack(([0], (var_set_dim ** 2).cumsum()[:self.num_var_set - 1]))

            gpu_begin_time = time()
            joint_logp = np.empty(self.N, dtype = np.float32)
            d_means = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = np.hstack(means).flatten().astype(np.float32))
            d_states = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = states.astype(np.int32))
            d_trans_p = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = trans_p.astype(np.float32))
            d_cov_dets = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.array([np.linalg.det(cov) for cov in covs], dtype=np.float32))
            d_cov_invs = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.hstack([np.linalg.inv(cov).flatten() for cov in covs]).astype(np.float32))
            d_var_set_dim = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = var_set_dim.astype(np.float32))
            d_var_set_linear_offset = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = var_set_linear_offset.astype(np.float32))
            d_var_set_sq_offset = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = var_set_sq_offset.astype(np.float32))
            
            d_joint_logp = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = joint_logp)

            self.cl_prg.calc_joint_logp(self.queue, (self.N, self.num_var_set), None,
                                        self.d_obs, d_states, d_trans_p, d_means, d_cov_dets, d_cov_invs,
                                        d_var_set_dim, d_var_set_linear_offset, d_var_set_sq_offset, d_joint_logp,
                                        np.int32(self.num_states), np.int32(self.num_var_set), np.int32(self.num_var), np.int32(self.num_cov_var))

            cl.enqueue_copy(self.queue, joint_logp, d_joint_logp)
            self.gpu_time += time() - gpu_begin_time

        else:
            joint_logp = np.empty(self.N, dtype=np.float32)
            
            # calculate transition probabilities first
            joint_logp[0] = trans_p[0, states[0]]
            joint_logp[1:] = np.log(trans_p[states[:self.N-1], states[1:]])

            # then emission probs
            for var_set_idx in xrange(self.num_var_set):
                var_set = self.obs_vars[var_set_idx]
                for state in self.uniq_states:
                    indices = np.where(states == state)
                    obs_set = np.array(self.data[var_set])[indices]
                    joint_logp[indices] += multivariate_normal.logpdf(obs_set,
                                                                      mean = means[state-1][var_set_idx],
                                                                      cov = covs[state-1][var_set_idx])
        return joint_logp.sum()

    def do_inference(self, str output_folder = None):
        """Perform inference on parameters.
        """
        cdef list header, obs_vars
        cdef int i
        
        if output_folder is None:
            output_folder = self.source_dirname
        else:
            output_folder += '/'

        # set up output samples file and write the header
        self.sample_fp = gzip.open(output_folder + '{0}-gaussian-hmm-samples.csv.gz'.format(self.source_filename), 'w')
        header = ['iteration', 'loglik', 'dimension', 'state'] + ['mu_{0}'.format(_) for _ in self.flat_obs_vars]

        # temporary measure - list singletons
        obs_vars = [[_] if type(_) is str else _ for _ in self.obs_vars]
        header += ['cov_{0}_{1}'.format(*_) for _ in itertools.chain(*[itertools.product(*[_] * 2) for _ in obs_vars])]
        header += ['trans_p_from_bd', 'trans_p_to_bd']
        header += ['trans_p_to_{0:d}'.format(_) for _ in self.uniq_states]
        header += ['has_obs_{0:d}'.format(_) for _ in range(1, self.N + 1)]
        print(','.join(header), file = self.sample_fp)
        
        # start sampling
        begin_time = time()
        for i in xrange(1, self.sample_size+1):
            self.set_temperature(iteration = i)
            if self.cl_mode:
                new_states = self._infer_states()
                new_means, new_covs = self._infer_means_covs()
                new_trans_p = self._infer_trans_p()
            else:
                new_states = self._infer_states()
                new_means, new_covs = self._infer_means_covs()
                new_trans_p = self._infer_trans_p()

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

    
    def _save_sample(self, long iteration):
        """Save the means and covariance matrices from the current iteration.
        """
        cdef long state, state_idx
        cdef list row
        for state_idx in xrange(self.num_states):
            state = self.uniq_states[state_idx]
            row = [iteration, self.loglik, self.num_var, state]
            row += list(np.hstack(self.means[state-1]))
            row += list(np.hstack([np.ravel(_) for _ in self.covs[state-1]]))
            row += [self.trans_p_matrix[0, state], self.trans_p_matrix[state, 0]]
            row += list(np.ravel(self.trans_p_matrix[state, 1:]))
            row += list((self.states == state).astype(np.bool).astype(np.int0))
            print(','.join([str(_) for _ in row]), file = self.sample_fp)
                
        return
