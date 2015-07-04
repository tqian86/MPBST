/* 
   calculate the log emisission probability of an observation (obs),
   given the state (work-group dim) and its sufficient stats (means, covs)
*/
kernel void calc_emit_logp(global float *obs,
			   global float *means,
			   global float *cov_dets,
			   global float *cov_invs,
			   global float *emit_logp,
			   uint dim) {
  
  uint obs_idx = get_global_id(0); // get the index of the observation of interest
  uint state_idx = get_global_id(1); // get the index of the state
  uint K = get_global_size(1); // get the total number of states
  
  float logp = 0;
  logp += dim * log(2 * M_PI) + log(cov_dets[state_idx]);

  float mat_mul = 0.0f;
  float mat_inner;
  for (int i = 0; i < dim; i++) {
    mat_inner = 0.0f;
    for (int j = 0; j < dim; j++) {
      mat_inner += (obs[obs_idx * dim + j] - means[state_idx * dim + j]) * 
	cov_invs[state_idx * dim * dim + j * dim + i];
    }
    mat_mul += mat_inner * (obs[obs_idx * dim + i] - means[state_idx * dim + i]);
  }
  
  emit_logp[obs_idx * K + state_idx] = -0.5 * (logp + mat_mul);
  
}
