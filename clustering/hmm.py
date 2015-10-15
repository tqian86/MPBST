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

hmm.read_csv("./data/data.csv", obs_vars = ["obs"], seq_id = None, header = True, timestamp = None)

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

import MPBST
from MPBST import *
import itertools
import numpy as np
from scipy.stats import multivariate_normal
from collections import Counter
from time import time
from math import lgamma

def multivariate_t(mu, Sigma, df, size=None):
    '''
    Output:
    Produce size samples of d-dimensional multivariate t distribution
    Input:
    mu = mean (d dimensional numpy array or scalar)
    Sigma = scale matrix (dxd numpy array)
    df = degrees of freedom
    size = # of samples to produce
    '''
    d = len(Sigma)
    Z = np.random.multivariate_normal(np.zeros(d), Sigma, size)
    g = np.repeat(np.random.gamma(df/2, 2/df, size), d).reshape(Z.shape)
    return mu + Z/np.sqrt(g)


def wishart(df, Sigma, size=None):

    dim = len(Sigma)
    Z = np.random.multivariate_normal(np.zeros(dim), Sigma, df)
    return Z.T.dot(Z)

class HMMSampler(BaseSampler):

    def __init__(self, num_states, search = True, cl_mode = False, cl_device = None,
                 sample_size = 1000, search_tolerance = 100, annealing = False, debug_mumble = False):
        """Initialize the base HMM sampler.
        """
        BaseSampler.__init__(self,
                             search = search,
                             search_tolerance = search_tolerance,
                             cl_mode = cl_mode,
                             cl_device = cl_device,
                             sample_size = sample_size, 
                             annealing = annealing,
                             debug_mumble = debug_mumble)

        self.data = None
        self.num_states = num_states
        self.uniq_states = np.linspace(1, self.num_states, self.num_states).astype(np.int64)
        self.SEQ_BEGIN, self.SEQ_END = 1, 2
        self.str_output_lines = []
        self.seq_id = None
        self.group = None

    def __str__(self):
        """Return a readable string representation of the sampler.
        """
        s = '{0} with {1} states'.format(type(self).__name__, self.num_states)
        if self.data is None:
            return s + ' (no training data imported)'
        else:
            s += ' running on training data {0}'.format(self.source_filepath)
            if self.seq_id:
                s += 'with sequence id "{0}"'.format(self.seq_id)
            if self.group:
                s += ', group id "{0}" and {1} cluster(s)'.format(self.group, self.num_clusters)
            return s

    def __param_str__(self):
        s = '{0}-k{1}'.format(type(self).__name__, self.num_states)
        if self.data is not None:
            if self.seq_id:
                s += '-seqid_{0}'.format(self.seq_id)
            if self.group:
                s += '-group_{0}-clusters_{1}'.format(self.group, self.num_clusters)
        return s
        
    def read_csv(self, filepath, obs_vars = ['obs'], seq_id = None, timestamp = None, group = None, num_clusters = 1, header = True):
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

        - Individual observation sequences can be separated by specifying the "seq_id" argument. 
        The argument expects the name of the column whose values are labels of observation sequences.

        For instance, if "subject" is specified as the sequence identifier, and the data set

        outcome,subject
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
        BaseSampler.read_csv(self, filepath = filepath, obs_vars = obs_vars, header = header)

        # create timestamp using row number if no timestamp is supplied
        if timestamp is None:
            timestamp = '_time_stamp_'
            self.original_data['_time_stamp_'] = range(self.N)

        def _make_trans_p_matrix(num_states):
            trans_p_matrix = np.random.random((num_states+1, num_states+1))
            return trans_p_matrix / trans_p_matrix.sum(axis=1)
            
        # create multiple transition matrices if group is supplied
        if group is None:
            self.num_clusters = 1
            self.group_labels = [0] * self.N
        else:
            self.group = group
            self.group_labels = self.original_data[group].values
            self.num_clusters = num_clusters

        # we have the same number of trans matrices as the number of clusters
        self.trans_p_matrices = [_make_trans_p_matrix(self.num_states) for _ in xrange(self.num_clusters)]

        # get unique group labels and the number of groups
        self.group_label_set = np.unique(self.group_labels)
        self.num_groups = self.group_label_set.shape[0]
        
        # assign each group into a cluster randomly
        clusters = np.random.randint(0, self.num_clusters, size = self.num_groups)
        self.group_cluster_dict = dict(zip(self.group_label_set, clusters))
        # make a mask array that indicates the cluster membership of each *data point*
        self.cluster_mask = np.empty(self.N, dtype=np.int)
        for group_label in self.group_label_set:
            self.cluster_mask[(self.group_labels == group_label)] = self.group_cluster_dict[group_label]
            
        # create boundary markers if seq_id is supplied
        if seq_id is None:
            self.original_data.sort(columns = timestamp, inplace=True)
            self.data = self.original_data[obs_vars]
            self.boundary_mask = np.zeros(self.N, dtype=np.int32)
            self.boundary_mask[0] = self.SEQ_BEGIN
            self.boundary_mask[-1] = self.SEQ_END
        else:
            if seq_id not in self.original_data:
                print('The specified sequence ID "{0}" cannot be found in the data'.format(seq_id), file = sys.stderr)
                sys.exit(0)
            if timestamp not in self.original_data:
                print('The specified timestamp "{0}" cannot be found in the data'.format(timestamp), file = sys.stderr)
                sys.exit(0)
            self.seq_id = seq_id
            self.original_data.sort(columns = [seq_id, timestamp], inplace=True)
            self.data = self.original_data[obs_vars]
            self.boundary_mask = np.zeros(self.N, dtype=np.int32)
            rle_values, rle_lengths = zip(*[(k, len(list(g))) for k, g in itertools.groupby(self.original_data[seq_id])])
            boundary_begins = np.array(rle_lengths).cumsum() - rle_lengths
            boundary_ends = np.array(rle_lengths).cumsum() - 1
            self.boundary_mask[boundary_begins] = self.SEQ_BEGIN
            self.boundary_mask[boundary_ends] = self.SEQ_END
            
        self.states = np.random.randint(low = 1, high = self.num_states + 1, size = self.N).astype(np.int32)

        if self.cl_mode:
            self.d_obs = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.array(self.data, dtype=np.float32, order='C'))

        self.setup_sample_output(filepath)

    def _infer_trans_p(self, states):
        """Infer the transitional probabilities between states without OpenCL.
        """
        trans_p_matrices = [None] * len(self.trans_p_matrices)
        trans_p_matrix_shape = self.trans_p_matrices[0].shape
        for cluster in xrange(self.num_clusters):
            
            # copy the current transition prob matrix; this is needed for search mode
            new_trans_p_matrix = np.empty(trans_p_matrix_shape)
            new_trans_p_matrix[:] = self.trans_p_matrices[cluster][:]

            # retrieve bigram pairs belonging to this cluster            
            to_indices = np.where((self.cluster_mask == cluster) & (self.boundary_mask != self.SEQ_BEGIN))[0]
            from_indices = to_indices - 1
            
            # make bigram pairs for easier counting
            pairs = zip(states[from_indices], states[to_indices])

            # add also pairs made up by boundary marks 
            begin_states = states[(self.cluster_mask == cluster) & (self.boundary_mask == self.SEQ_BEGIN)]
            pairs.extend(zip([0] * begin_states.shape[0], begin_states))
            end_states = states[(self.cluster_mask == cluster) & (self.boundary_mask == self.SEQ_END)]
            pairs.extend(zip(end_states, [0] * end_states.shape[0]))
            
            pair_count = Counter(pairs)

            # calculate the new parameters
            for state_from in np.insert(self.uniq_states, 0, 0):
                count_from_state = [_[0] for _ in pairs].count(state_from)
                for state_to in np.insert(self.uniq_states, 0, 0):
                    new_trans_p_matrix[state_from, state_to] = (pair_count[(state_from, state_to)] + 1) / (count_from_state + (1 + self.num_states) * 1)

            new_trans_p_matrix[0, 0] = 0
            new_trans_p_matrix[0] = new_trans_p_matrix[0] / new_trans_p_matrix[0].sum()

            trans_p_matrices[cluster] = new_trans_p_matrix

        return trans_p_matrices
            
class GaussianHMMSampler(HMMSampler):

    def __init__(self, num_states, search = True, cl_mode = False, cl_device = None,
                 sample_size = 1000, search_tolerance = 100, annealing = False, debug_mumble = False):
        """Initialize the base HMM sampler.
        """
        HMMSampler.__init__(self, num_states = num_states,
                            search = search,
                            search_tolerance = search_tolerance,
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
        
    def __str__(self):
        """Return a readable string representation of the sampler.
        """
        return HMMSampler.__str__(self)

    def read_csv(self, filepath, obs_vars = ['obs'], seq_id = None, timestamp = None, group = None, num_clusters = 1, header = True):
        """Read data from a csv file and set up means and covariance matrices for the Gaussian
        generative model.
        """
        flat_obs_vars = list(np.hstack(obs_vars))
        self.obs_vars = obs_vars; self.flat_obs_vars = flat_obs_vars

        HMMSampler.read_csv(self, filepath, obs_vars = flat_obs_vars, seq_id = seq_id,
                            timestamp = timestamp, group = group, num_clusters = num_clusters, header = header)

        self.num_var = len(flat_obs_vars)
        self.num_cov_var = np.sum([len(_) ** 2 if type(_) is list else 1 for _ in obs_vars])
        self.num_var_set = len(obs_vars)
        self.means = [[np.zeros(len(_)) if type(_) is list else np.zeros(1) for _ in obs_vars] for s in xrange(self.num_states)]
        self.covs = [[np.eye(len(_)) if type(_) is list else np.eye(1) for _ in obs_vars] for s in xrange(self.num_states)]
        
        return

    def _gaussian_mu0(self, dim):
        return np.zeros((1, dim))

    def _gaussian_k0(self, dim):
        return 1

    def _wishart_T0(self, dim):
        return np.eye(dim)

    def _wishart_v0(self, dim):
        return dim

    def _infer_cluster(self, states, trans_p_matrices):
        """Infer the clusters of each group.
        """
        if self.num_clusters == 1: return self.group_cluster_dict, self.cluster_mask
        group_cluster_dict = copy.deepcopy(self.group_cluster_dict)
        cluster_mask = np.copy(self.cluster_mask)
        
        # change the cluster of a group affects the transition probability only
        # so we only need to calculate that
        for group_label in self.group_label_set:
            logp_grid = np.empty(self.num_clusters)
            old_cluster = self.group_cluster_dict[group_label]

            # retrieve bigram pairs belonging to this group
            to_indices = np.where((self.group_labels == group_label) & (self.boundary_mask != self.SEQ_BEGIN))[0]
            from_indices = to_indices - 1
            
            to_states = states[to_indices]
            from_states = states[from_indices]
            
            begin_states = states[(self.group_labels == group_label) & (self.boundary_mask == self.SEQ_BEGIN)]
            if len(begin_states) > 0:
                from_states = np.append(from_states, [0] * begin_states.shape[0])
                to_states = np.append(to_states, begin_states)
            end_states = states[(self.group_labels == group_label) & (self.boundary_mask == self.SEQ_END)]
            if len(end_states) > 0:
                from_states = np.append(from_states, end_states)
                to_states = np.append(to_states, [0] * end_states.shape[0])

            cluster_count = Counter(group_cluster_dict.values()) # count up first
            for candidate_cluster in xrange(self.num_clusters):

                # get the transition prob matrix corresponding to the target cluster
                trans_p_matrix = trans_p_matrices[candidate_cluster]
                
                # put the results into the first cell just as a placeholder
                logp_grid[candidate_cluster] = np.log(trans_p_matrix[from_states, to_states]).sum()
                
                # calculate the "prior"
                n = cluster_count[candidate_cluster]
                # don't count itself
                if old_cluster == candidate_cluster: n -= 1
                
                logp_grid[candidate_cluster] += np.log((n + 1.0) / (self.num_groups - 1 + self.num_clusters * 1))
                
            new_cluster = sample(a = range(self.num_clusters), p = lognormalize(logp_grid))

            # record the results
            group_cluster_dict[group_label] = new_cluster
            cluster_mask[(self.group_labels == group_label)] = new_cluster

        return group_cluster_dict, cluster_mask
            
    def _infer_states(self):
        """Infer the state of each observation without OpenCL.
        """
        # copy the states first, this is needed for search mode
        new_states = np.empty_like(self.states); new_states[:] = self.states[:]
        
        # set up grid for storing log probabilities
        state_logp = np.empty((self.N, self.num_states))
        emit_logp = np.empty((self.N, self.num_states))

        # emission probabilities can be calculated in one pass
        for var_set_idx in xrange(len(self.obs_vars)):
            var_set = self.obs_vars[var_set_idx]
            obs_set = np.array(self.data[var_set])
            for state in self.uniq_states:
                if var_set_idx == 0: emit_logp[:, state-1] = 0
                emit_logp[:,state-1] += multivariate_normal.logpdf(obs_set,
                                                                   mean = self.means[state-1][var_set_idx],
                                                                   cov = self.covs[state-1][var_set_idx])
       
        # state probabilities need to be interated over
        for nth in xrange(self.N):
            group_label = self.group_labels[nth]
            cluster = self.group_cluster_dict[group_label]
            trans_p_matrix = self.trans_p_matrices[cluster]
            is_beginning = self.boundary_mask[nth] == self.SEQ_BEGIN
            is_end = self.boundary_mask[nth] == self.SEQ_END

            
            if is_beginning:
                trans_prev_logp = np.log(trans_p_matrix[0, 1:])
            else:
                trans_prev_logp = np.log(trans_p_matrix[new_states[nth - 1], 1:])

            if is_end:
                trans_next_logp = np.log(trans_p_matrix[1:, 0])
            else:
                trans_next_logp = np.log(trans_p_matrix[1:, new_states[nth + 1]])

            state_logp[nth] = trans_prev_logp + trans_next_logp
            
            # resample state
            new_states[nth] = sample(a = self.uniq_states,
                                     p = lognormalize(x = state_logp[nth] + emit_logp[nth], temp = self.annealing_temp))
            
        return new_states

    def _infer_means_covs(self, states):
        """Infer the means of each hidden state without OpenCL.
        """
        new_means = [[np.zeros(len(_)) if type(_) is list else np.zeros(1) for _ in self.obs_vars] for s in xrange(self.num_states)]
        new_covs = [[np.eye(len(_)) if type(_) is list else np.eye(1) for _ in self.obs_vars] for s in xrange(self.num_states)]

        for var_set_idx in xrange(len(self.obs_vars)):
            var_set = self.obs_vars[var_set_idx]
            if type(var_set) is str: obs_dim = 1
            else: obs_dim = len(var_set)
            obs = np.array(self.data[var_set])

            for state in self.uniq_states:
                
                # get observations currently assigned to this state
                cluster_obs = obs[np.where(states == state)]
                n = cluster_obs.shape[0]
                # compute sufficient statistics
                if n == 0:
                    mu = np.zeros(obs_dim)
                    cluster_obs = np.zeros((1, obs_dim))
                else:
                    mu = cluster_obs.mean(axis = 0)

                obs_deviance = cluster_obs - mu
                mu0_deviance = mu - self._gaussian_mu0(dim = obs_dim)
                cov_obs = np.dot(obs_deviance.T, obs_deviance)
                cov_mu0 = np.dot(mu0_deviance.T, mu0_deviance)

                v_n = self._wishart_v0(obs_dim) + n
                k_n = self._gaussian_k0(obs_dim) + n
                T_n = self._wishart_T0(obs_dim) + cov_obs + cov_mu0 * self._gaussian_k0(obs_dim) * n / k_n

                # new mu is sampled from a multivariate t with the following parameters
                df = v_n - obs_dim + 1
                mu_n = (self._gaussian_k0(obs_dim) * self._gaussian_mu0(obs_dim) + n * mu) / k_n
                Sigma = T_n / (k_n * df)

                # means and covs are sampled independently since close solutions exist and
                # this reduces autocorrelation between samples
            
                # resample the new mean vector
                new_means[state-1][var_set_idx][:] = multivariate_t(mu = mu_n, Sigma = Sigma, df = df)
                # resample the covariance matrix
                new_covs[state-1][var_set_idx][:] = np.linalg.inv(wishart(Sigma = np.linalg.inv(T_n), df = v_n))
                
        # a hacky but surprisingly effective way to alleviate label switching
        m = np.array([new_means[_][0][0] for _ in xrange(self.num_states)])
        reindex = m.argsort()
        new_means = [new_means[i] for i in reindex]
        new_covs = [new_covs[i] for i in reindex]
        return new_means, new_covs

    def _logprob(self, sample):
        """Calculate the log probability of model and data.
        """
        means, covs, trans_p, states, group_cluster_dict, cluster_mask = sample

        if self.cl_mode:
            var_set_dim = np.array([len(_) if type(_) is list else 1 for _ in self.obs_vars])
            var_set_linear_offset = np.hstack(([0], var_set_dim.cumsum()[:self.num_var_set - 1]))
            var_set_sq_offset = np.hstack(([0], (var_set_dim ** 2).cumsum()[:self.num_var_set - 1]))

            gpu_begin_time = time()
            joint_logp = np.empty((self.N, self.num_var_set), dtype=np.float32)
            d_means = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                hostbuf = np.hstack([np.hstack(_) for _ in means]).astype(np.float32))
            d_states = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = states.astype(np.int32))
            d_trans_p = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = trans_p.astype(np.float32))
            d_cov_dets = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.hstack([[np.linalg.det(_) for _ in cov] for cov in covs]).astype(np.float32))
            d_cov_invs = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.hstack([np.hstack([np.linalg.inv(_).ravel() for _ in cov]) for cov in covs]).astype(np.float32))
            d_var_set_dim = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = var_set_dim.astype(np.float32))
            d_var_set_linear_offset = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = var_set_linear_offset.astype(np.float32))
            d_var_set_sq_offset = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = var_set_sq_offset.astype(np.float32))
            
            d_joint_logp = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = joint_logp)
            self.cl_prg.calc_joint_logp(self.queue, (self.N, self.num_var_set), None,
                                        self.d_obs, d_states, d_trans_p, d_means, d_cov_dets, d_cov_invs,
                                        d_var_set_dim, d_var_set_linear_offset, d_var_set_sq_offset, d_joint_logp,
                                        np.int32(self.num_states), np.int32(self.num_var), np.int32(self.num_cov_var))
            cl.enqueue_copy(self.queue, joint_logp, d_joint_logp)
            self.gpu_time += time() - gpu_begin_time

        else:
            gpu_time = time()
            logprob_model = 0
            joint_logp = np.zeros(self.N)

            # calculate the probability of the clustering arrangement
            logprob_model += lgamma(self.num_clusters * 1) - lgamma(self.num_groups + self.num_clusters * 1)
            cluster_count = Counter(group_cluster_dict.values())
            for cluster in xrange(self.num_clusters):
                n = cluster_count[cluster]
                logprob_model += lgamma(n + 1) - lgamma(1)
            
            # calculate transition probabilities first
            for cluster in xrange(self.num_clusters):

                # retrieve bigram pairs belonging to this cluster
                to_indices = np.where((cluster_mask == cluster) & (self.boundary_mask != self.SEQ_BEGIN))[0]
                from_indices = to_indices - 1

                to_states = states[to_indices]
                from_states = states[from_indices]
                
                # add also pairs made up by boundary marks 
                begin_states = states[(cluster_mask == cluster) & (self.boundary_mask == self.SEQ_BEGIN)]
                from_states = np.append(from_states, [0] * begin_states.shape[0]).astype(np.int)
                to_states = np.append(to_states, begin_states).astype(np.int)
                end_states = states[(cluster_mask == cluster) & (self.boundary_mask == self.SEQ_END)]
                from_states = np.append(from_states, end_states).astype(np.int)
                to_states = np.append(to_states, [0] * end_states.shape[0]).astype(np.int)

                # put the result as logprob_model
                logprob_model += np.log(trans_p[cluster][from_states, to_states]).sum()
                
            # then emission probs
            for var_set_idx in xrange(len(self.obs_vars)):
                var_set = self.obs_vars[var_set_idx]
                for state in self.uniq_states:
                    indices = np.where(states == state)
                    obs_set = np.array(self.data[var_set])[indices]
                    joint_logp[indices] += multivariate_normal.logpdf(obs_set,
                                                                      mean = means[state-1][var_set_idx],
                                                                      cov = covs[state-1][var_set_idx])
            self.gpu_time += time() - gpu_time
        return logprob_model, joint_logp.sum()

    def do_inference(self, output_folder = None):
        """Perform inference on parameters.
        """
        if output_folder is None:
            output_folder = self.source_dirname
        else:
            output_folder += '/'

        # set up output samples file and write the header
        self.sample_fp = gzip.open(self.sample_fn, 'w')
        header = ['iteration', 'logprob_model', 'loglik_data', 'dimension', 'state'] + ['mu_{0}'.format(_) for _ in self.flat_obs_vars]

        # temporary measure - list singletons
        obs_vars = [[_] if type(_) is str else _ for _ in self.obs_vars]
        header += ['cov_{0}_{1}'.format(*_) for _ in itertools.chain(*[itertools.product(*[_] * 2) for _ in obs_vars])]
        for cluster in xrange(self.num_clusters):
            header += ['tp_from_bd_{0}'.format(cluster), 'tp_to_bd_{0}'.format(cluster)]
            header += ['tp_to_{0:d}_{1}'.format(_, cluster) for _ in self.uniq_states]
        for group_label in self.group_label_set:
            header += ['{0}_cluster'.format(group_label)]
        header += ['has_obs_{0:d}'.format(_) for _ in range(1, self.N + 1)]
        self.sample_fp.write(','.join(header) + '\n')
        
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
                new_means, new_covs = self._infer_means_covs(new_states)
                new_trans_p = self._infer_trans_p(new_states)
                new_group_cluster_dict, new_cluster_mask = self._infer_cluster(new_states, new_trans_p)
                
            if self.search:
                if self.better_sample((new_means, new_covs, new_trans_p, new_states, new_group_cluster_dict, new_cluster_mask)):
                    self._save_sample(iteration = i)
                if self.no_improvement(): break
                self.means, self.covs, self.trans_p_matrices, self.states = new_means, new_covs, new_trans_p, new_states
                self.group_cluster_dict, self.cluster_mask = new_group_cluster_dict, new_cluster_mask
            else:
                self.means, self.covs, self.trans_p_matrices, self.states = new_means, new_covs, new_trans_p, new_states
                self.group_cluster_dict, self.cluster_mask = new_group_cluster_dict, new_cluster_mask
                self.logprob_model, self.loglik_data = self._logprob((new_means, new_covs, new_trans_p, new_states,
                                                                      new_group_cluster_dict, new_cluster_mask))
                self._save_sample(iteration = i)
                
        self.sample_fp.close()
        self.total_time += time() - begin_time
    
        return self.gpu_time, self.total_time

    def _save_sample(self, iteration):
        """Save the means and covariance matrices from the current iteration.
        """
        for state in self.uniq_states:
            row = [iteration, round(self.logprob_model, 3), round(self.loglik_data, 3), self.num_var, state]
            row += list(np.hstack(np.round(self.means[state-1], 3)))
            row += list(np.hstack([np.ravel(_) for _ in np.round(self.covs[state-1], 3)]))
            for cluster in xrange(self.num_clusters):
                row += [round(self.trans_p_matrices[cluster][0, state], 3), round(self.trans_p_matrices[cluster][state, 0], 3)]
                row += list(np.ravel(np.round(self.trans_p_matrices[cluster][state, 1:], 3)))
            for group_label in self.group_label_set:
                row += [self.group_cluster_dict[group_label]]
            row += list((self.states == state).astype(np.bool).astype(np.int0))
            self.sample_fp.write(','.join([str(_) for _ in row]) + '\n')
            
        return

if __name__ == '__main__':
    hs = GaussianHMMSampler(num_states = 2, sample_size = 2000, search = False, cl_mode=False, debug_mumble = True)
    hs.read_csv('./toydata/speed.csv.gz', obs_vars = ['rt'], group='corr')
    gt, tt = hs.do_inference()
    print(gt, tt)
