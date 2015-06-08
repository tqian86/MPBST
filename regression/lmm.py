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
        self.params = pd.DataFrame(None, index = range(self.niter), columns = [])
        
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
                    self.params.loc[:,'beta_{0}_{1}'.format(predictor, gv)] = 0
        return
    
    def do_inference(self, output_file=None):
        for i in xrange(1, self.niter):
            self.infer_fixed(iter = i, output_file = output_file)
        return

    def infer_fixed(self, iter, output_file=None):
        """Infer the fixed effects.
        """
        for p in self.predictors:
            # check if p has random effects
            if p in self.predictor_groups:
                self._infer_fixed_w_random(p, iter, output_file)
            else:
                self._infer_fixed_wo_random(p, iter, output_file)

    def _infer_fixed_wo_random(self, p, iter, output_file = None):
        """Infer the beta coefficient of a fixed effect that has no random effects.
        """
        return

    def _infer_fixed_w_random(self, p, iter, output_file = None):
        """Infer the beta coefficient of a fixed effect that has random effects.
        """
        proposal_sd = .1
        old_beta = self.params.loc[iter-1, 'beta_%s' % p]
        new_beta = random.gauss(mu = old_beta, sigma = proposal_sd)

        # set up to calculate the g densities for both the old and new beta values
        log_g_old = 0 #-1 * old_beta # which is np.log(np.exp(-1 * old_beta))
        log_g_new = 0 #-1 * new_beta # similar as above

        k_0, mu_0, alpha_0, beta_0 = 0.1, 0, 1, 1

        # retrieve all groups where this predictor has random effects
        for group in self.predictor_groups[p]:
            random_betas = self.params.loc[iter-1, ['beta_%s_%s' % (p, gv) for gv in self.group_values[group]]]

            x_bar, x_var = random_betas.mean(), random_betas.var()
            n = len(random_betas)
            k_n = k_0 + n
            mu_n = (k_0 * mu_0 + n * x_bar) / k_n
            alpha_n = alpha_0 + n / 2
            beta_n = beta_0 + 0.5 * x_var * n + k_0 * n * (x_bar - mu_0) ** 2 / (2 * k_n)
            Lambda = alpha_n * k_n / beta_n
            
            t_frozen = t(df = 2 * alpha_n, loc = mu_n, scale = (1 / Lambda) ** 0.5)
            
            log_g_old += t_frozen.logpdf(old_beta)
            log_g_new += t_frozen.logpdf(new_beta)
            
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
                                    
    
lmm = LMMSampler(record_best = True, cl_mode = False)
lmm.read_csv('./data/10group-100n.csv')
lmm.set_fixed(outcome = 'y', predictors = [1, 'x1', 'x2'])
lmm.set_random(ranef_dict = {'group': [1, 'x1']})
lmm.do_inference()
print(lmm.params['beta_1'])
