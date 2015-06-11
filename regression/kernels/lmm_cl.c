kernel void resid_squares(global float *outcome,
			  global float *x,
			  global float *beta,
			  global float *resid2,
			  uint num_of_predictors) {
  uint id = get_global_id(0);
  float predicted_outcome = 0;
  for (uint i = 0; i < num_of_predictors; i++) {
    predicted_outcome += beta[i] * x[id * num_of_predictors + i];
  }
  resid2[id] = pow(outcome[id] - predicted_outcome, 2);

}

kernel void infer_fixed_beta(global float *resid_sum_new,
			     global float *rand,
			     global int *change,
			     float log_g_old,
			     float sigma2) {

  uint id = get_global_id(0);
  
  // set up to calculate the g densities for both the old and new beta values
  log_g_old += 0; // -1 * old_beta # which is np.log(np.exp(-1 * old_beta))
  float log_g_new = 0; // -1 * new_beta # similar as above
        
  // modify the beta value and calculate the new loglikelihood
  log_g_new += - (1 / (2 * sigma2)) * resid_sum_new[id];

  // compute candidate densities q for old and new beta
  // since the proposal distribution is normal this step is not needed
  float log_q_old = 0; //np.log(dnorm(old_beta, loc = new_beta, scale = proposal_sd))
  float log_q_new = 0; //np.log(dnorm(new_beta, loc = old_beta, scale = proposal_sd)) 
        
  //compute the moving probability
  float moving_logprob = (log_g_new + log_q_old) - (log_g_old + log_q_new);
        
  change[id] = log(rand[id]) < moving_logprob;
}
