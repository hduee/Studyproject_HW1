data {
  int<lower=0> N;
  vector[N] semantic_similarity; // semantic similarity between word and context
  vector[N] phonological_similarity; // phonological similarity between word and context
  vector[N] response_time;
}

transformed data {
  vector[N] log_rt = log(response_time);
}

parameters {
  real alpha;
  real beta_sem;
  real beta_phon;
  real<lower=0> sigma_rt;
}

model {
  alpha ~ normal(0, 5);
  beta_sem ~ normal(0, 2);
  beta_phon ~ normal(0, 2);
  sigma_rt ~ normal(0, 1) T[0,];

  log_rt ~ normal(alpha + beta_sem * semantic_similarity + beta_phon * phonological_similarity, sigma_rt);
}

generated quantities {
  vector[N] log_likelihood;
  vector[N] log_rt_ppd;
  vector[N] rt_ppd;

  for (i in 1:N) {
    log_likelihood[i] = normal_lpdf(log_rt[i] | alpha
                                                + beta_sem * semantic_similarity[i]
                                                + beta_phon * phonological_similarity[i], sigma_rt);

    // posterior predictive draws
    log_rt_ppd[i] = normal_rng(alpha
                               + beta_sem * semantic_similarity[i]
                               + beta_phon * phonological_similarity[i], sigma_rt);
    rt_ppd[i] = exp(log_rt_ppd[i]);
  }
}