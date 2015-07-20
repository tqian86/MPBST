/*
static float sum(global float *arr, int start, int length) {
  float result = 0;
  for (int i = start; i < start + length; i++) {
    result += arr[i];
  }
  return result;
}

static float max_arr(global float *arr, int start, int length) {
  float result = arr[start];
  for (int i = start + 1; i < start + length; i++) {
    //if (arr[i] > result) result = arr[i];
    result = fmax(result, arr[i]);
  }
  return result;
}

static void lognormalize(global float *logp, int start, int length) {
  float m = max_arr(logp, start, length);
  for (int i = start; i < start + length; i++) {
    logp[i] = powr(exp(1.0f), logp[i] - m);
    // this line is a hack to prevent a weird nVIDIA-only global memory bug
    // !logp is global
    // a simple exp(logp[i] - m) fails to compile on nvidia cards
  }
  float p_sum = sum(logp, start, length);
  for (int i = start; i < start + length; i++) {
    logp[i] = logp[i] / p_sum;
  }
}


static uint sample(uint a_size, global uint *a, global float *p, int start, float rand) {
  float total = 0.0f;
  for (uint i = start; i < start + a_size; i++) {
    total = total + p[i];
    if (total > rand) return a[i-start];
  }
  return a[a_size - 1];
}
*/

/* 
   calculate the log emisission probability of an observation (obs),
   given the state (work-group dim) and its sufficient stats (means, covs)
*/
/*
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
  logp += dim * log(2 * M_PI_F) + log(cov_dets[state_idx]);

  // calcualte (x - mu).T %*% inv(cov) %*% (x - mu)
  float mat_mul = 0.0f;
  float mat_inner;
  for (uint i = 0; i < dim; i++) {
    mat_inner = 0.0f;
    for (uint j = 0; j < dim; j++) {
      mat_inner += (obs[obs_idx * dim + j] - means[state_idx * dim + j]) * 
	cov_invs[state_idx * dim * dim + j * dim + i];
    }
    mat_mul += mat_inner * (obs[obs_idx * dim + i] - means[state_idx * dim + i]);
  }
  
  emit_logp[obs_idx * K + state_idx] = -0.5f * (logp + mat_mul);
  
}
*/
/*
  calculate the log state probability of each state
*/
/*
kernel void calc_state_logp(global int *states,
			    global float *trans_p_matrix,
			    global float *state_logp,
			    uint obs_idx, uint N) {

  int state_idx = get_global_id(0);
  int num_states = get_global_size(0);

  float trans_prev_logp = (obs_idx != 0) * log(trans_p_matrix[(states[obs_idx-1 * (obs_idx!=0)]-1) * num_states + state_idx]) +
    (obs_idx == 0) * log(1.0f / num_states);

  float trans_next_logp = (obs_idx != N - 1) * log(trans_p_matrix[state_idx * num_states + (states[obs_idx + 1 * (obs_idx != N -1)] - 1)]) +
    (obs_idx == N - 1) * log(1.0f);

  state_logp[obs_idx * num_states + state_idx] = trans_prev_logp + trans_next_logp;
  
}
*/
/* resample states */
/*
kernel void resample_state(global int *states,
			   global float *state_logp,
			   global float *emit_logp,
			   global float *temp_logp,
			   global float *rand,
			   uint obs_idx, uint num_states) {

  for (uint i = 0; i < num_states; i++) {
    temp_logp[i] = state_logp[obs_idx * num_states + i] + emit_logp[obs_idx * num_states + i];
  }

  lognormalize(temp_logp, 0, num_states);

  float total = 0.0f;
  for (uint i = 0; i < num_states; i++) {
    total += temp_logp[i];
    if (total > rand[obs_idx]) {
      states[obs_idx] = i+1;
      return;
    }
  }
  states[obs_idx] = num_states;
}
*/
/* 
   calculate the log joint probability of an hmm model, which incldues
   the emisission probability of an observation (obs), and the transi-
   tional probabilities between states
*/
kernel void calc_joint_logp(global float *obs,
			    global int *states,
			    global float *trans_p,
			    global float *means,
			    global float *cov_dets,
			    global float *cov_invs,
			    global float *var_set_dim,
			    global float *var_set_linear_offset,
			    global float *var_set_sq_offset,
			    global float *joint_logp,
			    uint num_state, uint num_var_set, uint num_var, uint num_cov_var) {
  
  uint obs_idx = get_global_id(0); // get the index of the observation of interest
  uint state_idx = states[obs_idx] - 1;

  uint vs_idx = get_global_id(1); // get the index of the variable set
  uint vs_dim = var_set_dim[vs_idx]; // dimension of the variable set
  uint vs_offset = var_set_linear_offset[vs_idx];
  uint vs_sq_offset = var_set_sq_offset[vs_idx];
  
  float logp = 0;
  logp += vs_dim * log(2 * M_PI_F) + log(cov_dets[state_idx * num_var_set + vs_idx]);

  // calcualte (x - mu).T %*% inv(cov) %*% (x - mu)
  float mat_mul = 0.0f;
  float mat_inner;
  for (uint i = 0; i < vs_dim; i++) {
    mat_inner = 0.0f;
    for (uint j = 0; j < vs_dim; j++) {
      mat_inner += (obs[obs_idx * num_var + vs_offset + j] - means[state_idx * num_var + vs_offset + j]) * 
	cov_invs[state_idx * num_cov_var + vs_sq_offset + j * vs_dim + i];
    }
    mat_mul += mat_inner * (obs[obs_idx * num_var + vs_offset + i] - means[state_idx * num_var + vs_offset + i]);
  }
  logp = -0.5f * (logp + mat_mul);

  // calculate transitional probabilities
  uint prev_state = (obs_idx > 0) * states[(obs_idx - 1) * (obs_idx > 0)]; // this will be 0 for the first observation
  logp = logp + log(trans_p[prev_state * (num_state + 1) + (state_idx + 1)]);
  joint_logp[obs_idx] = logp;
}
