functions {
  row_vector softmax_log(row_vector log_probs) {
    return exp(log_probs - log_sum_exp(log_probs));
  }
}

data {
  int<lower=1> C;                       // number of contexts (trials)
  int<lower=1> U;                       // number of utterances
  int<lower=1> S_max;                   // maximum number of objects per context
  array[C] int<lower=1, upper=S_max> n_states;
  array[C, S_max] int<lower=0, upper=1> size_extension;
  array[C, S_max] int<lower=0, upper=1> color_extension;

  vector[U] cost;

  int<lower=1> N;                       // number of observed utterances
  array[N] int<lower=1, upper=C> context_id;
  array[N] int<lower=1, upper=U> utterance;

  array[C] int<lower=1, upper=S_max> target_index;
  int<lower=1, upper=U> redundant_index;

  int<lower=1> T;                       // number of target objects (for aggregation)
  array[C] int<lower=1, upper=T> target_group;
  array[T] int<lower=1> target_totals;
}

parameters {
  real<lower=0, upper=1> alpha_raw;
  real<lower=0, upper=1> cost_weight_raw;

  real<lower=0, upper=1> size_semantics_raw;
  real<lower=0, upper=1> color_semantics_raw;
}

transformed parameters {
  real alpha = 0.1 + 40.0 * alpha_raw;
  real cost_weight = 0.01 + 5.0 * cost_weight_raw;

  real size_semantics = 0.001 + 0.998 * size_semantics_raw;
  real color_semantics = 0.001 + 0.998 * color_semantics_raw;

  array[C] vector[S_max] meaning_size;
  array[C] vector[S_max] meaning_color;

  array[C] matrix[U, S_max] L0;
  array[C] matrix[S_max, U] S1;

  for (c in 1:C) {
    for (s in 1:n_states[c]) {
      meaning_size[c][s] = size_extension[c, s] * size_semantics
                           + (1 - size_extension[c, s]) * (1 - size_semantics);
      meaning_color[c][s] = color_extension[c, s] * color_semantics
                            + (1 - color_extension[c, s]) * (1 - color_semantics);
    }
    if (n_states[c] < S_max) {
      for (s in (n_states[c] + 1):S_max) {
        meaning_size[c][s] = 0;
        meaning_color[c][s] = 0;
      }
    }

    for (u in 1:U) {
      row_vector[n_states[c]] log_probs;
      for (s in 1:n_states[c]) {
        real meaning = (u == 1)
                         ? meaning_size[c][s]
                         : (u == 2)
                           ? meaning_color[c][s]
                           : meaning_size[c][s] * meaning_color[c][s];
        log_probs[s] = log(meaning) - log(n_states[c]);
      }
      row_vector[n_states[c]] l0_slice = softmax_log(log_probs);
      for (s in 1:n_states[c]) {
        L0[c][u, s] = l0_slice[s];
      }
      if (n_states[c] < S_max) {
        for (s in (n_states[c] + 1):S_max) {
          L0[c][u, s] = 0;
        }
      }
    }

    for (s in 1:n_states[c]) {
      row_vector[U] log_utility;
      for (u in 1:U) {
        log_utility[u] = alpha * log(L0[c][u, s]) - cost_weight * cost[u];
      }
      row_vector[U] s1_slice = softmax_log(log_utility);
      for (u in 1:U) {
        S1[c][s, u] = s1_slice[u];
      }
    }
    if (n_states[c] < S_max) {
      for (s in (n_states[c] + 1):S_max) {
        for (u in 1:U) {
          S1[c][s, u] = 0;
        }
      }
    }
  }
}

model {
  for (n in 1:N) {
    int ctx = context_id[n];
    int target_state = target_index[ctx];
    row_vector[U] speaker_row = S1[ctx][target_state];
    target += categorical_lpmf(utterance[n] | to_vector(speaker_row));
  }
}

generated quantities {
  vector[N] log_lik;
  vector[C] overinform_prob_context;
  vector[T] overinform_prob_target;
  array[T] int overinform_rep;

  for (n in 1:N) {
    int ctx = context_id[n];
    int target_state = target_index[ctx];
    row_vector[U] speaker_row = S1[ctx][target_state];
    log_lik[n] = categorical_lpmf(utterance[n] | to_vector(speaker_row));
  }

  for (c in 1:C) {
    overinform_prob_context[c] = S1[c][target_index[c], redundant_index];
  }

  for (t in 1:T) {
    real sum_prob = 0;
    int count = 0;
    for (c in 1:C) {
      if (target_group[c] == t) {
        sum_prob += overinform_prob_context[c];
        count += 1;
      }
    }
    overinform_prob_target[t] = sum_prob / count;
    overinform_rep[t] = binomial_rng(target_totals[t], overinform_prob_target[t]);
  }
}
