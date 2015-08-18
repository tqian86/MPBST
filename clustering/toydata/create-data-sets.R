## This script creates several toy data sets for the test of various clustering models
## such as the Hidden Markov Model

## Data Set 1:
##
## This data set is generated from a hidden markov model whose edges are annotated with
## "group" information. That is, transitions between two neighboring states can belong
## to different groups and be described by multiple transition matrices.
##
## Such a hidden markov model can be used to capture the notion of "eventuality" - that
## State i might transition to State j given a certain period of time, but will eventually
## go to State k.

## Generate the states
num.states = 2

trans.p.matrices = list(
    matrix(c(0, 0.02, 0.02, 0.5, 0.49, 0.49, 0.5, 0.49, 0.49), ncol = 3),
    matrix(c(0, 0.02, 0.02, 0.5, 0.29, 0.19, 0.5, 0.69, 0.79), ncol = 3),
    matrix(c(0, 0.02, 0.02, 0.5, 0.19, 0.09, 0.5, 0.79, 0.89), ncol = 3)
)

sampleNextState <- function(current = 0, group, trans.p.matrices) {
    trans.p.matrix <- trans.p.matrices[[group]]
    sample(x = 1:num.states, size = 1, prob = trans.p.matrix[current + 1,2:(num.states+1)])
}

N = 5000
mu = c(1, 10)
sd = c(1, 2)

group.seq = sample(1:3, size = N, replace = T)
state.seq = c()
obs.seq = c()
for (i in 1:N) {
    if (i == 1) {
        state = sampleNextState(group = group.seq[i], trans.p.matrices = trans.p.matrices)
        state.seq = c(state.seq, state)
        obs.seq = c(obs.seq, rnorm(n = 1, mean = mu[state], sd = sd[state]))
    } else {
        state = sampleNextState(current = state.seq[i-1],
                                group = group.seq[i],
                                trans.p.matrices = trans.p.matrices)
        state.seq = c(state.seq, state)
        obs.seq = c(obs.seq, rnorm(n = 1, mean = mu[state], sd = sd[state]))
    }
}

d1 = data.frame(state = as.factor(state.seq),
                group = as.factor(group.seq),
                y = obs.seq)

write.csv(d1, gzfile('mtm-hmm.csv.gz'), row.names=F)
