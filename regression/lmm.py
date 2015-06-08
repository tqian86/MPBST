#!/usr/bin/env python
#!-*- coding: utf-8 -*-

from __future__ import print_function, division

import sys, os.path
pkg_dir = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(pkg_dir)

from MPBST import *
from scipy.stats import t

class LMMSampler(RegressionSampler):

    def __init__(self, record_best = True, cl_mode = False, cl_device = None):
        """Initialize the class.
        """
        RegressionSampler.__init__(self, record_best, cl_mode, cl_device)
        self.outcome = None
        self.predictors = []
        self.group_predictors = {}
        self.group_values = {}
        self.predictor_groups = {}
        self.params = pd.DataFrame(1.0, index = range(self.niter), columns = ['sigma2'])
        
    def set_fixed(self, outcome, predictors):
        """Set the regression model in terms of a univariate outcome
        and a list of predictors.
        """
        self.outcome = outcome
        for p in predictors:
            if p == 1 or p in self.obs:
                self.predictors.append(p)
                self.params.loc[:,'beta_%s' % p] = 0
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
                # store valid predictors - group to predictors
                try: self.group_predictors[group_name].append(predictor)
                except: self.group_predictors[group_name] = [predictor]
                # reverse hash - predictor to groups
                try: self.predictor_groups[predictor].append(group_name)
                except: self.predictor_groups[predictor] = [group_name]
                # create parameter storage
                for gv in group_values:
                    self.params.loc[:,'beta_{0}_{1}{2}'.format(predictor, group_name, gv)] = 0
                self.params.loc[:,'sigma2_{0}_{1}'.format(predictor, group_name)] = 1.0
        return
    
    def do_inference(self, output_file=None):
        for iter in xrange(1, self.niter):

            # Infer the fixed effects.
            for p in self.predictors:
                self._infer_fixed_beta(p, iter, output_file)
                
            # Infer overall model variance sigma2
            self._infer_sigma2(iter, output_file = None)

    def _infer_fixed_beta(self, p, iter, output_file = None):
        """Infer the beta coefficient of a fixed effect that has random effects.
        """
        proposal_sd = .2
        old_beta = self.params.loc[iter-1, 'beta_%s' % p]
        new_beta = random.gauss(mu = old_beta, sigma = proposal_sd)

        # set up to calculate the g densities for both the old and new beta values
        log_g_old = 0 #-1 * old_beta # which is np.log(np.exp(-1 * old_beta))
        log_g_new = 0 #-1 * new_beta # similar as above
        
        Beta = self.params.filter(regex = '^beta').loc[iter-1]
        X = self.obs.filter(items=[colname.split('_')[1] for colname in Beta.index])

        # calculate the old loglikelihood
        log_g_old += - (1 / (2 * self.params.loc[iter-1, 'sigma2'])) * \
                     np.sum((self.obs[self.outcome] - np.dot(X, Beta)) ** 2)

        # modify the beta value and calculate the new loglikelihood
        Beta.loc['beta_%s' % p] = new_beta
        log_g_new += - (1 / (2 * self.params.loc[iter-1, 'sigma2'])) * \
                     np.sum((self.obs[self.outcome] - np.dot(X, Beta)) ** 2)

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
                                    

    def _infer_sigma2(self, iter, output_file = None):
        """Infer the variance of the overall multi-level model.
        """
        alpha_0 = 1
        beta_0 = 1
        alpha_n = alpha_0 + self.N / 2

        Beta = self.params.filter(regex = '^beta').loc[iter-1]
        X = self.obs.filter(items=[colname.split('_')[1] for colname in Beta.index])

        beta_n = beta_0 + 0.5 * np.sum((self.obs[self.outcome] - np.dot(X, Beta)) ** 2)
        self.params.loc[iter, 'sigma2'] = np.random.gamma(alpha_n, 1 / beta_n)
        
lmm = LMMSampler(record_best = True, cl_mode = False)
lmm.read_csv('./data/10group-100n.csv')
lmm.set_fixed(outcome = 'y', predictors = [1, 'x1', 'x2'])
lmm.set_random(ranef_dict = {'group': [1, 'x1']})
lmm.do_inference()
print(lmm.params.iloc[900:1000])
