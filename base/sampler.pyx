#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
cimport numpy as np
import pandas as pd
import sys, copy, random, math, csv, gzip, mimetypes, os.path
from time import time

def smallest_unused_label(int_labels):

    if len(int_labels) == 0: return [], [], 0
    label_count = np.bincount(int_labels)
    try:
        new_label = np.where(label_count == 0)[0][0]
    except IndexError:
        new_label = max(int_labels) + 1
    uniq_labels = np.unique(int_labels)
    return label_count, uniq_labels, new_label

def lognormalize(x, float temp = 1):
    """Normalize a vector of logprobabilities to probabilities that sum up to 1.
    Optionally accepts an annealing temperature that does simple annealing.
    """
    cdef np.ndarray x_nparr = np.array(x)
    x_nparr = x_nparr - np.max(x_nparr)
    # anneal
    cdef np.ndarray xp = np.power(np.exp(x_nparr), temp)
    return xp / xp.sum()

def sample(a, p):
    """Step sample from a discrete distribution using CDF
    """
    if (len(a) != len(p)):
        raise Exception('a != p')
    cdef np.ndarray p_arr = np.array(p)
    p_arr = p_arr / p_arr.sum()
    cdef float r = random.random()
    cdef int n = len(a)
    cdef float total = 0           # range: [0,1]
    for i in xrange(n):
        total += p_arr[i]
        if total > r:
            return a[i]
    return a[i]

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
    cdef np.ndarray Z = np.random.multivariate_normal(np.zeros(dim), Sigma, df)
    return Z.T.dot(Z)

def sample_wishart(sigma, df):
    '''
    Returns a sample from the Wishart distn, conjugate prior for precision matrices.
    '''
    
    n = sigma.shape[0]
    chol = np.linalg.cholesky(sigma)
    
    # use matlab's heuristic for choosing between the two different sampling schemes
    if (df <= 81+n) and (df == round(df)):
    # direct
        X = np.dot(chol,np.random.normal(size=(n,df)))
    else:
        A = np.diag(np.sqrt(np.random.chisquare(df - np.arange(0,n),size=n)))
        A[np.tri(n,k=-1,dtype=bool)] = np.random.normal(size=(n*(n-1)/2.))
        X = np.dot(chol,A)
        
    return np.dot(X,X.T)

def print_matrix_in_row(npmat, file_dest):
    """Print a matrix in a row.
    """
    row, col = npmat.shape
    print(col, *npmat.reshape((1, row * col))[0], sep=',', file=file_dest)
    return True

class BaseSampler(object):

    def __init__(self, record_best, cl_mode, cl_device = None, niter=1000, thining = 0, annealing = False, debug_mumble = False):
        """Initialize the class.
        """
        if cl_mode:
            import pyopencl as cl
            import pyopencl.array, pyopencl.tools, pyopencl.clrandom
            if cl_device == 'gpu':
                gpu_devices = []
                for platform in cl.get_platforms():
                    try: gpu_devices += platform.get_devices(device_type=cl.device_type.GPU)
                    except: pass
                self.ctx = cl.Context(gpu_devices)
            elif cl_device == 'cpu':
                cpu_devices = []
                for platform in cl.get_platforms():
                    try: cpu_devices += platform.get_devices(device_type=cl.device_type.CPU)
                    except: pass
                self.ctx = cl.Context([cpu_devices[0]])
            else:
                self.ctx = cl.create_some_context()

            self.queue = cl.CommandQueue(self.ctx)
            self.mem_pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.queue))
            self.mf = cl.mem_flags
            self.device = self.ctx.get_info(cl.context_info.DEVICES)[0]
            self.device_type = self.device.type
            self.device_compute_units = self.device.max_compute_units

        self.cl_mode = cl_mode
        self.obs = []
        self.niter = niter
        self.thining = thining
        self.burnin = 0
        self.N = 0 # number of data points
        self.best_sample = (None, None) # (sample, loglikelihood)
        self.record_best = record_best
        self.best_diff = []
        self.no_improv = 0
        self.gpu_time = 0
        self.total_time = 0
        self.header_written = False
        self.annealing = annealing
        self.annealing_temp = 1
        self.debug_mumble = debug_mumble
        
    def read_csv(self, filepath, header = True):
        """Read data from a csv file.
        """
        # determine if the type file is gzip
        filetype, encoding = mimetypes.guess_type(filepath)
        if encoding == 'gzip':
            csvfile = gzip.open(filepath, 'r')
        else:
            csvfile = open(filepath, 'r')

        #dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        reader = csv.reader(csvfile)#, dialect)
        if header:
            reader.next()
        for row in reader:
            self.obs.append([_ for _ in row])

        self.N = len(self.obs)
        return

    def direct_read_obs(self, obs):
        self.obs = obs

    def set_temperature(self, iteration):
        """Set the temperature of simulated annealing as a function of sampling progress.
        """
        if self.annealing is False:
            self.annealing_temp = 1.0
            return

        if iteration < self.niter * 0.2:
            self.annealing_temp = 0.2
        elif iteration < self.niter * 0.3:
            self.annealing_temp = 0.4
        elif iteration < self.niter * 0.4:
            self.annealing_temp = 0.6
        elif iteration < self.niter * 0.5:
            self.annealing_temp = 0.8
        else:
            self.annealing_temp = 1.0

    def do_inference(self, output_file = None):
        """Perform inference. This method does nothing in the base class.
        """
        return

    def auto_save_sample(self, sample):
        """Save the given sample as the best sample if it yields
        a larger log-likelihood of data than the current best.
        """
        cdef float new_logprob = self._logprob(sample)
        # if there's no best sample recorded yet
        if self.best_sample[0] is None and self.best_sample[1] is None:
            self.best_sample = (sample, new_logprob)
            if self.debug_mumble: print('Initial sample generated, loglik: {0}'.format(new_logprob), file=sys.stderr)
            return

        # if there's a best sample
        if new_logprob > self.best_sample[1]:
            self.no_improv = 0
            self.best_diff.append(new_logprob - self.best_sample[1])
            self.best_sample = (copy.deepcopy(sample), new_logprob)
            if self.debug_mumble: print('New best sample found, loglik: {0}'.format(new_logprob), file=sys.stderr)
            return True
        else:
            self.no_improv += 1
            return False

    def no_improvement(self, threshold=500):
        if len(self.best_diff) == 0: return False
        if self.no_improv > threshold or np.mean(self.best_diff[-threshold:]) < .1:
            print('Too little improvement in loglikelihood for %s iterations - Abort searching' % threshold, file=sys.stderr)
            return True
        return False

    def _logprob(self, sample):
        """Compute the logliklihood of data given a sample. This method
        does nothing in the base class.
        """
        return

class RegressionSampler(BaseSampler):

    def read_csv(self, filepath):
        """Read data from a csv file and store them in a numpy array.
        Create a header index that maps array column indices to
        variable names.
        """
        self.obs = pd.read_csv(filepath)
        self.obs.loc[:,'1'] = 1
        self.N = self.obs.shape[0]

        