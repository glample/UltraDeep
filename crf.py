import theano
import theano.tensor as T


def log_sum_exp(x, axis=None):
    """
    Sum probabilities in the log-space.
    """
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))


def forward(observations, transitions, log_space=True, viterbi=False,
            return_alpha=False):
    """
    Takes as input:
        - observations, sequence of shape (n_steps, n_classes)
        - transitions, sequence of shape (n_classes, n_classes)
    Probabilities can be given in the log space.
    Compute alpha, matrix of size (n_steps, n_classes), such that
    alpha[i, j] represents one of these 2 values:
        - the probability that the real path at node i is in j
        - the maximum probability of a path finishing in j at node i (Viterbi)
    Returns one of these 2 values:
        - alpha
        - the final probability, which can be:
            - the sum of the probabilities of all the paths
            - the probability of the best path (Viterbi)
    """
    def recurrence(obs, previous, transitions):
        previous = previous.dimshuffle(0, 'x')
        obs = obs.dimshuffle('x', 0)
        if viterbi:
            out = (previous + obs + transitions).max(axis=0)
        else:
            if log_space:
                out = log_sum_exp(previous + obs + transitions, axis=0)
            else:
                out = (previous + obs + transitions).sum(axis=0)
        return out

    # assert observations.ndim == 2
    # assert transitions.ndim == 2

    initial = observations[0]
    alpha, _ = theano.scan(
        fn=recurrence,
        outputs_info=initial,
        sequences=[observations[1:]],
        non_sequences=transitions
    )

    if return_alpha:
        return alpha
    else:
        if viterbi:
            return alpha[-1].max(axis=0)
        else:
            return log_sum_exp(alpha[-1], axis=0)


def forward_dynamic(observations, transitions, log_space=True, viterbi=False,
                    return_alpha=False):
    """
    Takes as input:
        - observations, sequence of shape (n_steps, n_classes)
        - transitions, sequence of shape (n_classes, n_classes, n_classes)
    Probabilities can be given in the log space.
    Compute alpha, matrix of size (n_steps, n_classes), such that
    alpha[i, j] represents one of these 2 values:
        - the probability that the real path at node i is in j
        - the maximum probability of a path finishing in j at node i (Viterbi)
    Returns one of these 2 values:
        - alpha
        - the final probability, which can be:
            - the sum of the probabilities of all the paths
            - the probability of the best path (Viterbi)
    """
    def recurrence(obs, previous, transitions):
        previous = previous.dimshuffle(0, 'x')
        obs = obs.dimshuffle('x', 0)
        local_transitions = transitions.dot(obs)
        if viterbi:
            out = (previous + obs + local_transitions).max(axis=0)
        else:
            if log_space:
                out = log_sum_exp(previous + obs + local_transitions, axis=0)
            else:
                out = (previous + obs + local_transitions).sum(axis=0)
        return out

    # assert observations.ndim == 2
    # assert transitions.ndim == 3

    initial = observations[0]
    alpha, _ = theano.scan(
        fn=recurrence,
        outputs_info=initial,
        sequences=[observations[1:]],
        non_sequences=transitions
    )

    if return_alpha:
        return alpha
    else:
        if viterbi:
            return alpha[-1].max(axis=0)
        else:
            return log_sum_exp(alpha[-1], axis=0)
