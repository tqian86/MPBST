kernel void dotProd(global float *result, global float *m1,  global float *m2, uint m1_width) {
 
  int m1_height = get_global_size(0);
  int m2_width = get_global_size(1);
  int i = get_global_id(0);
  int j = get_global_id(1);

  int k;
  float temp_r = 0.0f;
  for (k = 0; k < m1_width; k++) {
    temp_r += m1[i * m1_width + k] * m2[j + k * m2_width];
    //if (k < 3) printf("i: %d, j: %d, k: %d %f %f\n", i, j, k, m1[i * m1_width + k], m2[k * m2_width + j]);
  }
  result[i * m2_width + j] = temp_r;
}

kernel void dotProd2(global float *result, global float *m1,  global float *m2, uint q, uint r) {
 
  int p = get_global_size(0);
  int i = get_global_id(0);

  int k;
  float temp_r;

  for (int j = 0; j < r; j++) {
    temp_r = 0.0f;
    for (k = 0; k < q; k++) {
      temp_r += m1[i * q + k] * m2[j + k * r];
      //printf("i: %d, j: %d, k: %d %f %f\n", i, j, k, m1[i * q + k], m2[k * r + j]);
    }
    result[i * r + j] = temp_r;
  }
}
