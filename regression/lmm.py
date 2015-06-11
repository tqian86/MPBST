#!/usr/bin/env python
#!-*- coding: utf-8 -*-

from __future__ import print_function, division

import sys, os.path
pkg_dir = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(pkg_dir)

from MPBST import *

class LMMSampler(RegressionSampler):

    def __init__(self, record_best = True, cl_mode = False, cl_device = None, niter = 1000):
        """Initialize the class.
        """
        RegressionSampler.__init__(self, record_best, cl_mode, cl_device, niter)

        if cl_mode:
            program_str = open(pkg_dir + 'MPBST/regression/kernels/lmm_cl.c', 'r').read()
            util_str = open(pkg_dir + 'MPBST/kernels/utilities_cl.c', 'r').read()
            self.cl_prg = cl.Program(self.ctx, program_str).build()
            self.cl_util = cl.Program(self.ctx, util_str).build()
            self.d_X, self.d_outcome_obs, self.resid2 = None, None, None
        
        self.outcome = None
        self.outcome_obs = None
        self.predictors = []
        self.group_values = {}
        self.predictor_groups = {}
        self.params = pd.DataFrame(1.0, index = range(self.niter), columns = ['sigma2'])
        self.X = None

    def set_fixed(self, outcome, predictors):
        """Set the regression model in terms of a univariate outcome
        and a list of predictors.
        """
        self.outcome = outcome
        self.outcome_obs = self.obs[outcome]
        if self.cl_mode:
            self.d_outcome_obs = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                           hostbuf = np.array(self.outcome_obs, dtype=np.float32, order='C'))
        
        for p in predictors:
            if p == 1 or p in self.obs:
                self.predictors.append(p)
                self.params.loc[:,'beta_%s' % p] = random.random()
            else:
                print('Predictor %s not found in the data. Ignored' % p,
                      file = sys.stderr)
        
    def set_random(self, ranef_dict):
        """Set the random effects in the form of dictionaries.
        For example: {"subject": [1, "age"]} indicates subject-level
        random intercepts and slopes for the predictor 'age'.
        """
        for group_name, predictors in ranef_dict.iteritems():
            # make sure the group exists
            assert group_name in self.obs
            # get the unique values of group identifiers
            group_values = set(self.obs[group_name])
            self.group_values[group_name] = group_values
            
            for predictor in predictors:
                # make sure there's a fixed effect for this predictor
                assert predictor in self.predictors
                # reverse hash - predictor to groups
                try: self.predictor_groups[predictor].append(group_name)
                except: self.predictor_groups[predictor] = [group_name]
                # create parameter storage
                for gv in group_values:
                    self.params.loc[:,'beta_{0}_{1}{2}'.format(predictor, group_name, gv)] = 0
                self.params.loc[:,'sigma2_{0}_{1}'.format(predictor, group_name)] = 1.0
        return
    
    def do_inference(self, output_file=None):
        """Perform inference on the parameters.
        """
        begin_time = time()
        # construct an X matrix that would readily multiple with all coefficients
        beta_coef_names = self.params.filter(regex = '^beta').columns
        self.X = copy.deepcopy(self.obs.filter(items=[colname.split('_')[1] for colname in beta_coef_names]))
        self.X.columns = ['_'.join(colname.split('_')[1:]) for colname in beta_coef_names]
        # set cells not corresponding to the appropriate groups to 0
        for g in self.group_values.iterkeys():
            for gv in self.group_values[g]:
                self.X.at[self.obs[g] == gv, self.X.columns.to_series().str.contains('.*%s(?!%s$)' % (g, gv))] = 0
        
        if self.cl_mode:
            for iter in xrange(1, self.niter):

                gpu_begin_time = time()
                Beta = copy.deepcopy(self.params.filter(regex = '^beta')).loc[iter-1]
                d_Beta = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = np.array(Beta).astype(np.float32))

                if self.d_X is None:
                    self.d_X = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = np.array(self.X, dtype = np.float32, order='C'))

                
                resid2 = np.empty(shape = self.N, dtype=np.float32)
                d_resid2 = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = resid2)
                self.cl_prg.resid_squares(self.queue, (self.N,), None,
                                          self.d_outcome_obs, self.d_X, d_Beta, d_resid2, np.int32(Beta.shape[0]))
                cl.enqueue_copy(self.queue, resid2, d_resid2)
                self.gpu_time += time() - gpu_begin_time
                log_g_old = - (1 / (2 * self.params.at[iter-1, 'sigma2'])) * resid2.sum()
                
                # Infer the fixed effects
                old_fixed_Beta = np.array([Beta.at['beta_%s' % p] for p in self.predictors], dtype=np.float32)
                new_fixed_Beta = np.random.normal(old_fixed_Beta, scale=0.05).astype(np.float32)
                rand = np.random.random(size = new_fixed_Beta.shape).astype(np.float32)
                change = np.empty(shape = new_fixed_Beta.shape, dtype = np.int32)
                resid_sum_new = np.empty(shape = new_fixed_Beta.shape, dtype = np.float32)

                for i in xrange(new_fixed_Beta.shape[0]):
                    Beta_temp = copy.deepcopy(Beta)
                    Beta_temp.loc['beta_%s' % self.predictors[i]] = new_fixed_Beta[i]
                    d_Beta_temp = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = np.array(Beta_temp, dtype=np.float32, order='C'))
                    self.cl_prg.resid_squares(self.queue, (self.N,), None,
                                              self.d_outcome_obs, self.d_X, d_Beta_temp, d_resid2, np.int32(Beta.shape[0]))
                    cl.enqueue_copy(self.queue, resid2, d_resid2)
                    resid_sum_new[i] = resid2.sum()

                gpu_begin_time = time()
                d_rand = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = rand)
                d_resid_sum_new = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = resid_sum_new)
                d_change = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = change)
                    
                self.cl_prg.infer_fixed_beta(self.queue, new_fixed_Beta.shape, None,
                                             d_resid_sum_new, d_rand, d_change, np.float32(log_g_old), np.float32(self.params.at[iter-1, 'sigma2']))
                cl.enqueue_copy(self.queue, change, d_change)
                self.gpu_time += time() - gpu_begin_time
                
                for i in xrange(new_fixed_Beta.shape[0]):
                    if change[i] == 1: self.params.loc[iter, 'beta_%s' % self.predictors[i]] = new_fixed_Beta[i]
                    else: self.params.loc[iter, 'beta_%s' % self.predictors[i]] = old_fixed_Beta[i]

                # Infer overall model variance sigma2
                self._cl_infer_sigma2(iter, Beta, self.X, output_file)

                # Infer the random effects
                for p, groups in self.predictor_groups.iteritems():
                    for g in groups:
                        for gv in self.group_values[g]:
                            self._infer_random_beta(p, g, gv, iter, Beta, self.X, log_g_old,  output_file)

                        # Infer the variance of each random beta
                        self._infer_random_sigma2(p, g, iter, Beta, self.X, output_file)

                if self.record_best:
                    self.auto_save_sample(self.params.iloc[iter])
                    if self.no_improvement(300):
                        break

        else:
            for iter in xrange(1, self.niter):
            
                Beta = copy.deepcopy(self.params.filter(regex = '^beta')).loc[iter-1]
            
                # calculate the old loglikelihood - this can be reused
                log_g_old = - (1 / (2 * self.params.at[iter-1, 'sigma2'])) * \
                            np.sum((self.outcome_obs - np.dot(self.X, Beta)) ** 2)

                # Infer the fixed effects
                for p in self.predictors:
                    self._infer_fixed_beta(p, iter, Beta, self.X, log_g_old, output_file)

                # Infer overall model variance sigma2
                self._infer_sigma2(iter, Beta, self.X, output_file)

                # Infer the random effects
                for p, groups in self.predictor_groups.iteritems():
                    for g in groups:
                        for gv in self.group_values[g]:
                            self._infer_random_beta(p, g, gv, iter, Beta, self.X, log_g_old,  output_file)

                        # Infer the variance of each random beta
                        self._infer_random_sigma2(p, g, iter, Beta, self.X, output_file)

                if self.record_best:
                    self.auto_save_sample(self.params.iloc[iter])
                    if self.no_improvement(300):
                        break
                    
        self.total_time += time() - begin_time

        return self.gpu_time, self.total_time

    def _infer_fixed_beta(self, p, iter, Beta, X, log_g_old, output_file = None):
        """Infer the beta coefficient of a fixed effect that has random effects.
        """
        Beta_temp = copy.deepcopy(Beta)
        old_beta = Beta.at['beta_%s' % p]
        proposal_sd = 0.05
        new_beta = random.gauss(mu = old_beta, sigma = proposal_sd)

        # set up to calculate the g densities for both the old and new beta values
        log_g_old += 0 #-1 * old_beta # which is np.log(np.exp(-1 * old_beta))
        log_g_new = 0 #-1 * new_beta # similar as above
        
        # modify the beta value and calculate the new loglikelihood
        Beta_temp.at['beta_%s' % p] = new_beta
        log_g_new += - (1 / (2 * self.params.at[iter-1, 'sigma2'])) * \
                     np.sum((self.outcome_obs - np.dot(X, Beta_temp)) ** 2)

        # compute candidate densities q for old and new beta
        # since the proposal distribution is normal this step is not needed
        log_q_old = 0#np.log(dnorm(old_beta, loc = new_beta, scale = proposal_sd))
        log_q_new = 0#np.log(dnorm(new_beta, loc = old_beta, scale = proposal_sd)) 
        
        # compute the moving probability
        moving_prob = min(1, np.exp((log_g_new + log_q_old) - (log_g_old + log_q_new)))
        
        u = random.uniform(0,1)
        if u < moving_prob:
            self.params.loc[iter, 'beta_%s' % p] = new_beta
            return True
        else:
            self.params.loc[iter, 'beta_%s' % p] = old_beta
            return False

        
    def _infer_sigma2(self, iter, Beta, X, output_file = None):
        """Infer the variance of the overall multi-level model.
        """
        alpha_0 = 1
        beta_0 = 1
        alpha_n = alpha_0 + self.N / 2
        beta_n = beta_0 + 0.5 * np.sum((self.outcome_obs - np.dot(X, Beta)) ** 2)
        self.params.at[iter, 'sigma2'] = 1 / np.random.gamma(alpha_n, 1 / beta_n)

    def _cl_infer_sigma2(self, iter, Beta, X, output_file = None):
        """Infer the variance of the overall multi-level model.
        """
        alpha_0 = 1
        beta_0 = 1
        alpha_n = alpha_0 + self.N / 2
        gpu_begin_time = time()
        d_Beta = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = np.array(Beta).astype(np.float32))
        resid2 = np.empty(shape = self.X.shape[0], dtype=np.float32)
        d_resid2 = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = resid2)
        self.cl_prg.resid_squares(self.queue, (self.N,), None,
                                      self.d_outcome_obs, self.d_X, d_Beta, d_resid2, np.int32(Beta.shape[0]))
        cl.enqueue_copy(self.queue, resid2, d_resid2)
        self.gpu_time += time() - gpu_begin_time
        
        beta_n = beta_0 + 0.5 * resid2.sum()
        self.params.at[iter, 'sigma2'] = 1 / np.random.gamma(alpha_n, 1 / beta_n)
        
    def _infer_random_beta(self, p, g, gv, iter, Beta, X, log_g_old, output_file = None):
        """Infer the beta coefficient of a fixed effect that has random effects.
        """
        Beta_temp = copy.deepcopy(Beta)
        proposal_sd = .01
        sigma2_eff = self.params.at[iter-1, 'sigma2_{0}_{1}'.format(p, g)]
        old_beta = Beta.loc['beta_%s_%s%s' % (p, g, gv)]
        new_beta = random.gauss(mu = old_beta, sigma = proposal_sd)

        # prior is the expectation that beta ~ N(0, sigma2_g)
        log_g_old += - (1 / (2 * sigma2_eff)) * old_beta ** 2
        log_g_new = - (1 / (2 * sigma2_eff)) * new_beta ** 2
        
        
        # modify the beta value and calculate the new loglikelihood
        Beta_temp.loc['beta_%s_%s%s' % (p, g, gv)] = new_beta
        log_g_new += - (1 / (2 * self.params.at[iter-1, 'sigma2'])) * \
                     ((self.outcome_obs - np.dot(X, Beta_temp)) ** 2).sum()

        # compute candidate densities q for old and new beta
        # since the proposal distribution is normal this step is not needed
        log_q_old = 0#np.log(dnorm(old_beta, loc = new_beta, scale = proposal_sd))
        log_q_new = 0#np.log(dnorm(new_beta, loc = old_beta, scale = proposal_sd)) 
        
        # compute the moving probability
        moving_prob = min(1, np.exp((log_g_new + log_q_old) - (log_g_old + log_q_new)))
        
        u = random.uniform(0,1)
        if u < moving_prob:
            self.params.loc[iter, 'beta_%s_%s%s' % (p, g, gv)] = new_beta
            return True
        else:
            self.params.loc[iter, 'beta_%s_%s%s' % (p, g, gv)] = old_beta
            return False

    def _cl_infer_random_beta(self, p, g, gv, iter, Beta, X, log_g_old, output_file = None):
        """Infer the beta coefficient of a fixed effect that has random effects.
        """
        proposal_sd = .01
        sigma2_eff = self.params.at[iter-1, 'sigma2_{0}_{1}'.format(p, g)]
        old_beta = Beta.loc['beta_%s_%s%s' % (p, g, gv)]
        new_beta = random.gauss(mu = old_beta, sigma = proposal_sd)

        # prior is the expectation that beta ~ N(0, sigma2_g)
        log_g_old += - (1 / (2 * sigma2_eff)) * old_beta ** 2
        log_g_new = - (1 / (2 * sigma2_eff)) * new_beta ** 2
        
        
        # modify the beta value and calculate the new loglikelihood
        Beta.loc['beta_%s_%s%s' % (p, g, gv)] = new_beta
        gpu_begin_time = time()
        d_Beta = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = np.array(Beta).astype(np.float32))
        resid2 = np.empty(shape = self.X.shape[0], dtype=np.float32)
        d_resid2 = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = resid2)
        self.cl_prg.resid_squares(self.queue, (self.N,), None,
                                      self.d_outcome_obs, self.d_X, d_Beta, d_resid2, np.int32(Beta.shape[0]))
        cl.enqueue_copy(self.queue, resid2, d_resid2)
        self.gpu_time += time() - gpu_begin_time

        log_g_new += - (1 / (2 * self.params.at[iter-1, 'sigma2'])) * resid2.sum()

        # compute candidate densities q for old and new beta
        # since the proposal distribution is normal this step is not needed
        log_q_old = 0#np.log(dnorm(old_beta, loc = new_beta, scale = proposal_sd))
        log_q_new = 0#np.log(dnorm(new_beta, loc = old_beta, scale = proposal_sd)) 
        
        # compute the moving probability
        moving_prob = min(1, np.exp((log_g_new + log_q_old) - (log_g_old + log_q_new)))
        
        u = random.uniform(0,1)
        if u < moving_prob:
            self.params.loc[iter, 'beta_%s_%s%s' % (p, g, gv)] = new_beta
            return True
        else:
            self.params.loc[iter, 'beta_%s_%s%s' % (p, g, gv)] = old_beta
            return False

        
    def _infer_random_sigma2(self, p, g, iter, Beta, X, output_file = None):
        """Infer the variance of sub-level  models.
        """
        alpha_0 = 1
        beta_0 = 1
        n = self.params.filter(regex='beta_{0}_{1}.+$'.format(p, g)).shape[1]
        alpha_n = alpha_0 + n / 2
        beta_n = beta_0 + 0.5 * np.sum((self.params.filter(regex='beta_{0}_{1}'.format(p, g)).loc[iter-1] - 0) ** 2)
        self.params.at[iter, 'sigma2_{0}_{1}'.format(p, g)] = 1 / np.random.gamma(alpha_n, 1 / beta_n)


    def _logprob(self, sample):
        """Calculate the loglikelihood of data given a sample.
        """
        Beta = copy.deepcopy(sample.filter(regex = '^beta'))
        logprob =  - self.N / 2 * np.log(2 * math.pi * sample['sigma2']) + \
                   (- ((self.outcome_obs - np.dot(self.X, Beta)) ** 2).sum() / (2 * sample['sigma2']))
        for p, groups in self.predictor_groups.iteritems():
            for g in groups:
                n = sample.filter(regex='beta_{0}_{1}.+$'.format(p, g)).shape[0]
                logprob += - n / 2 * np.log(2 * math.pi * sample['sigma2_{0}_{1}'.format(p, g)]) + \
                    (- ((sample.filter(regex='beta_{0}_{1}'.format(p, g)) - 0) ** 2).sum() / (2 * sample['sigma2_{0}_{1}'.format(p, g)]))
        return logprob
        
        
lmm = LMMSampler(record_best = True, cl_mode = False, niter=1000)
lmm.read_csv('./data/10group-100n.csv')
lmm.set_fixed(outcome = 'y', predictors = [1, 'x1', 'x2'])
lmm.set_random(ranef_dict = {'group': ['x1', 'x2']})
gpu_time, total_time = lmm.do_inference()
print("OpenCL device time: %f seconds; Total_time: %f seconds\n" % (gpu_time, total_time), file=sys.stderr)

#lmm.params.to_csv('samples.csv')
print(lmm.best_sample)
