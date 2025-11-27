functions {
  row_vector softmax_log(row_vector log_probs) {
    return exp(log_probs - log_sum_exp(log_probs));
  }
}

data {
  int<lower=1> S;
  int<lower=1> U;
  int<lower=1> N;             // number of trials

  matrix<lower=0,upper=1>[U, S] meaning_matrix;

  array[U] int<lower=0, upper=1> size_indicator;
  array[U] int<lower=0, upper=1> color_indicator;

  array[N] int<lower=1, upper=S> state_id; // which state on trial n
  array[N] int<lower=1, upper=U> utt_id; // which utterance was produced on trial n
}

parameters {
  real<lower=0> alpha;
  real<lower=0> cost_size;
  real<lower=0> cost_color;
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
        real utterance_cost = cost_size * size_indicator[u] + cost_color * color_indicator[u];
        log_utility[u] = alpha * log(L0[u, s]) - utterance_cost;
    }
    S1[s] = softmax_log(log_utility);
  }
}

model {
  alpha ~ normal(0, 5);
  cost_size ~ normal(0, 5);
  cost_color ~ normal(0, 5);

  // likelihood: utterances are categorical with probs S1[state]
  for (n in 1:N) {
    utt_id[n] ~ categorical(S1[state_id[n]]');
  }
}

generated quantities {
  array[N] int<lower=1, upper=U> utt_rep; // replicated utterances
  vector[N] log_lik;

  for (n in 1:N) {
    // simulate a new utterance for the same state
    utt_rep[n] = categorical_rng(S1[state_id[n]]');

    // pointwise log-likelihood for model comparison
    log_lik[n] = categorical_lpmf(utt_id[n] | S1[state_id[n]]');
  }
}
