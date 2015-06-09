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
        self.outcome_obs = None
        self.predictors = []
        self.group_values = {}
        self.predictor_groups = {}
        self.params = pd.DataFrame(1.0, index = range(self.niter), columns = ['sigma2'])
        self.proposal_sd = .1
        
    def set_fixed(self, outcome, predictors):
        """Set the regression model in terms of a univariate outcome
        and a list of predictors.
        """
        self.outcome = outcome
        self.outcome_obs = self.obs[outcome]
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
                # reverse hash - predictor to groups
                try: self.predictor_groups[predictor].append(group_name)
                except: self.predictor_groups[predictor] = [group_name]
                # create parameter storage
                for gv in group_values:
                    self.params.loc[:,'beta_{0}_{1}{2}'.format(predictor, group_name, gv)] = 0
                self.params.loc[:,'sigma2_{0}_{1}'.format(predictor, group_name)] = 1.0
        return
    
    def do_inference(self, output_file=None):
        X = None
        for iter in xrange(1, self.niter):
            
            Beta = copy.deepcopy(self.params.filter(regex = '^beta')).loc[iter-1]
            if X is None:
                X = copy.deepcopy(self.obs.filter(items=[colname.split('_')[1] for colname in Beta.index]))
                X.columns = ['_'.join(colname.split('_')[1:]) for colname in Beta.index]
                for g in self.group_values.iterkeys():
                    for gv in self.group_values[g]:
                        X.at[self.obs[g] == gv, X.columns.to_series().str.contains('.*%s(?!%s$)' % (g, gv))] = 0
            
            # calculate the old loglikelihood - this can be reused
            log_g_old = - (1 / (2 * self.params.loc[iter-1, 'sigma2'])) * \
                         np.sum((self.outcome_obs - np.dot(X, Beta)) ** 2)

            # Infer the fixed effects.
            for p in self.predictors:
                self._infer_fixed_beta(p, iter, Beta, X, log_g_old, output_file)
                
            # Infer overall model variance sigma2
            self._infer_sigma2(iter, Beta, X, output_file)

            # Infer the random effects
            for p, groups in self.predictor_groups.iteritems():
                for g in groups:
                    for gv in self.group_values[g]:
                        self._infer_random_beta(p, g, gv, iter, Beta, X, output_file)

                    self._infer_random_sigma2(p, g, iter, Beta, X, output_file)

    def _infer_fixed_beta(self, p, iter, Beta, X, log_g_old, output_file = None):
        """Infer the beta coefficient of a fixed effect that has random effects.
        """
        old_beta = Beta.at['beta_%s' % p]
        proposal_sd = self.proposal_sd
        new_beta = random.gauss(mu = old_beta, sigma = proposal_sd)

        # set up to calculate the g densities for both the old and new beta values
        log_g_old += 0 #-1 * old_beta # which is np.log(np.exp(-1 * old_beta))
        log_g_new = 0 #-1 * new_beta # similar as above
        
        # modify the beta value and calculate the new loglikelihood
        Beta.at['beta_%s' % p] = new_beta
        log_g_new += - (1 / (2 * self.params.loc[iter-1, 'sigma2'])) * \
                     np.sum((self.outcome_obs - np.dot(X, Beta)) ** 2)

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

    def _infer_random_beta(self, p, g, gv, iter, Beta, X, output_file = None):
        """Infer the beta coefficient of a fixed effect that has random effects.
        """
        proposal_sd = .025
        old_beta = Beta.loc['beta_%s_%s%s' % (p, g, gv)]
        new_beta = random.gauss(mu = old_beta, sigma = proposal_sd)

        # set up to calculate the g densities for both the old and new beta values
        log_g_old = 0 #-1 * old_beta # which is np.log(np.exp(-1 * old_beta))
        log_g_new = 0 #-1 * new_beta # similar as above

        log_g_old += - (1 / (2 * self.params.loc[iter-1, 'sigma2'])) * \
                     np.sum((self.outcome_obs[self.obs[g] == gv] - np.dot(X[self.obs[g] == gv], Beta)) ** 2)
        log_g_old += - (1 / (2 * self.params.loc[iter-1, 'sigma2_{0}_{1}'.format(p, g)])) * old_beta ** 2
        
        # modify the beta value and calculate the new loglikelihood
        Beta.loc['beta_%s_%s%s' % (p, g, gv)] = new_beta
        log_g_new += - (1 / (2 * self.params.loc[iter-1, 'sigma2'])) * \
                     np.sum((self.outcome_obs[self.obs[g] == gv] - np.dot(X[self.obs[g] == gv], Beta)) ** 2)
        log_g_new += - (1 / (2 * self.params.loc[iter-1, 'sigma2_{0}_{1}'.format(p, g)])) * new_beta ** 2

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
        alpha_n = alpha_0 + self.N / 2
        beta_n = beta_0 + 0.5 * np.sum((self.params.filter(regex='beta_{0}_{1}'.format(p, g)).loc[iter-1] - 0) ** 2)
        self.params.at[iter, 'sigma2_{0}_{1}'.format(p, g)] = 1 / np.random.gamma(alpha_n, 1 / beta_n)
        
        
lmm = LMMSampler(record_best = True, cl_mode = False)
lmm.read_csv('./data/10group-100n.csv')
lmm.set_fixed(outcome = 'y', predictors = [1, 'x1', 'x2'])
lmm.set_random(ranef_dict = {'group': ['x1', 'x2']})
lmm.do_inference()
lmm.params.to_csv('samples.csv')
