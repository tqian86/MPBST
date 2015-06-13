#!/usr/bin/env R
#!-*- coding: utf-8 -*-

d = read.csv('./data/10group-100n.csv')

lmm <- function(formula, data, niter = 1000) {
    require(data.table)
    # convert the data to data.table as well
    data = data.table(data)
    
    # process the formula
    f.terms = terms(formula)
    f.vars = attr(f.terms, "variables")
    if (attr(f.terms, "response") == 1) {
        outcome = as.character(f.vars[[2]])
    }
    intercept = attr(f.terms, "intercept")
    fixed = grep("[|]", attr(f.terms, "term.labels"), value=T, invert=T)

    # create the results matrix for storing samples
    samples = as.data.table(
        matrix(0,
               nrow = niter,
               ncol = intercept + length(fixed))
    )
    if (intercept == 1) {
        fixed = c('Intercept', fixed)
        data[, Intercept := 1]
    }
    setnames(samples, 1:ncol(samples), fixed)
    samples[, "sigma2" := 1]
    samples[1, x1 := 0.25]

    for (i in 2:niter) {
        inferFixedBeta(samples, data, i, fixed.vars = fixed, outcome = outcome)
    }
    samples
}

inferFixedBeta <- function(samples, data, iter, fixed.vars, outcome, proposal.sd = 0.05) {
    sigma2 = as.numeric(samples[iter-1, 'sigma2', with=F])
    outcome.obs = data[, outcome, with=F]
    X = as.matrix(data[, fixed.vars, with=F])
    Beta = samples[iter-1, fixed.vars, with=F]
    logrand = log(runif(n = length(fixed.vars)))
    for (vi in 1:length(fixed.vars)) {
        v = fixed.vars[vi]
        old.beta = as.numeric(Beta[,v,with=F])
        new.beta = rnorm(1, mean = old.beta, sd = proposal.sd)
        
        # calculate the old loglikelihood
        log.g.old = - (1 / (2 * sigma2)) *
            sum((outcome.obs - X %*% t(Beta)) ** 2)
       
        # modify the beta value and calculate the new loglikelihood
        Beta[, eval(v):= new.beta]
        log.g.new = - (1 / (2 * sigma2)) *
            sum((outcome.obs - X %*% t(Beta)) ** 2)
        
        # compute candidate densities q for old and new beta
        # since the proposal distribution is normal this step is not needed
        log.q.old = 0
        log.q.new = 0 
        
        # compute the moving probability
        moving.logprob = (log.g.new + log.q.old) - (log.g.old + log.q.new)
        samples[iter, eval(v) := ifelse(logrand[vi] < moving.logprob, new.beta, old.beta)]
    }
}
