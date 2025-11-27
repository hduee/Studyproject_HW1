functions {
  // A function that transforms log probabilities into normalized probabilities so all probabilities add to 1
  row_vector softmax_log(row_vector log_probs) {
    return exp(log_probs - log_sum_exp(log_probs));
  }
}

data {
  int<lower=1> S; // Number of states
  int<lower=1> U; // Number of utterances

  matrix<lower=0,upper=1>[U, S] meaning_matrix;   // Binary matrix that indicates if an utterance describes a state

  vector[U] cost;   // Production cost for each utterance

  real<lower=0> alpha; // Determines speaker rationality, how highly informativeness is valued
  real<lower=0> cost_weight; // Determines the impact of the cost vector 
}

parameters {
}

transformed parameters { 
  vector[S] state_prior = rep_vector(1.0 / S, S); // Uniform prior over all states
  vector[U] utterance_prior = rep_vector(1.0 / U, U); // Uniform prior over all utterances

  // Computes the matrix for the Literal Listener L0, the probability of that a state has been refered to given a specific utterance
  matrix[U, S] L0;
  for (u in 1:U) {
    row_vector[S] log_probs;
    for (s in 1:S) {
      // Apply Bayes rule
      log_probs[s] = log(meaning_matrix[u, s]) + log(state_prior[s]);
    }
    // Normalize probabilities
    L0[u] = softmax_log(log_probs);
  }

   // Computes the utility matrix for the Pragmatic Speaker S1, trading off between informativeness and production cost
  matrix[S, U] S1;
  for (s in 1:S) {
    row_vector[U] log_utility;
    for (u in 1:U) {
        log_utility[u] = alpha * log(L0[u, s]) - cost_weight * cost[u];
    }
    // Normalize probabilities
    S1[s] = softmax_log(log_utility);
  }
}

model {
}

generated quantities {
}