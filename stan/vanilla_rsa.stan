functions {
  row_vector softmax_log(row_vector log_probs) {
    return exp(log_probs - log_sum_exp(log_probs));
  }
}

data {
  int<lower=1> S;
  int<lower=1> U;

  matrix<lower=0,upper=1>[U, S] meaning_matrix;

  vector[U] cost;

  real<lower=0> alpha;
  real<lower=0> cost_weight;
}

parameters {
}

transformed parameters {
  vector[S] state_prior = rep_vector(1.0 / S, S);
  vector[U] utterance_prior = rep_vector(1.0 / U, U);

  matrix[U, S] L0;
  for (u in 1:U) {
    row_vector[S] log_probs;
    for (s in 1:S) {
      log_probs[s] = log(meaning_matrix[u, s]) + log(state_prior[s]);
    }
    L0[u] = softmax_log(log_probs);
  }

  matrix[S, U] S1;
  for (s in 1:S) {
    row_vector[U] log_utility;
    for (u in 1:U) {
        log_utility[u] = alpha * log(L0[u, s]) - cost_weight * cost[u];
    }
    S1[s] = softmax_log(log_utility);
  }
}

model {
}

generated quantities {
}