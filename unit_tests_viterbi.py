import numpy as np
import theano
import crf


# Sequences of observations with probabilities for each state
observations_test = [
    [[0.8, 0.2], [0.1, 0.9]],
    [[0.8, 0.2], [0.5, 0.2], [0.3, 0.6]],
    [[0.1, 0.6, 0.8], [0.3, 0.2, 0.1]]
]
observations_test = [np.array(x, dtype=np.float32) for x in observations_test]

# Transition probabilities for each sequence
transitions_test = [
    [[0.4, 0.6], [0.7, 0.3]],
    [[0.4, 0.6], [0.1, 0.9]],
    [[0.4, 0.4, 0.1], [0.3, 0.8, 0.3], [0.1, 0.1, 0.2]]
]
transitions_test = [np.array(x, dtype=np.float32) for x in transitions_test]

# Probabilities that a sequence ends at a particular state
alpha_last_test = [
    [(0.8 * 0.4 + 0.2 * 0.7) * 0.1, (0.8 * 0.6 + 0.2 * 0.3) * 0.9],
    [((0.8 * 0.4 + 0.2 * 0.1) * 0.5 * 0.4 + (0.8 * 0.6 + 0.2 * 0.9) * 0.2 * 0.1) * 0.3,
     ((0.8 * 0.4 + 0.2 * 0.1) * 0.5 * 0.6 + (0.8 * 0.6 + 0.2 * 0.9) * 0.2 * 0.9) * 0.6],
    [(0.1 * 0.4 + 0.6 * 0.3 + 0.8 * 0.1) * 0.3,
     (0.1 * 0.4 + 0.6 * 0.8 + 0.8 * 0.1) * 0.2,
     (0.1 * 0.1 + 0.6 * 0.3 + 0.8 * 0.2) * 0.1],
]
alpha_last_test = [np.array(x, dtype=np.float32) for x in alpha_last_test]

# Best path probability for each sequence
best_path_prob_test = [
    [0.8 * 0.6 * 0.9],
    [0.8 * 0.4 * 0.5 * 0.6 * 0.6],
    [0.6 * 0.8 * 0.2]
]
best_path_prob_test = [np.array(x, dtype=np.float32) for x in best_path_prob_test]


def log_sum_exp(x, axis=None):
    """
    Sum probabilities in the log-space.
    """
    x = np.asarray(x)
    xmax = x.max(axis=axis)
    if axis is not None:
        assert -x.ndim <= axis < x.ndim
        if axis < 0:
            axis += x.ndim
        idx_tuple = (
            [slice(None)] * axis +
            [None] +
            (x.ndim - axis - 1) * [slice(None)]
        )
    else:
        idx_tuple = Ellipsis
    return xmax + np.log(np.exp(x - xmax[idx_tuple]).sum(axis=axis))


def forward_np_slow(observations, transitions, viterbi=False,
                    return_alpha=False):
    """
    Takes as input:
        - observations, sequence of shape (n_steps, n_classes)
        - transitions, sequence of shape (n_classes, n_classes)
    Probabilities have to be given in the log space.
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
    n_steps, n_classes = observations.shape
    alpha = np.empty((n_steps, n_classes))
    alpha[...] = np.nan
    alpha[0, :] = observations[0:1]
    # Use maximum if we are doing Viterbi decoding, logaddexp otherwise.
    reducer = np.maximum if viterbi else np.logaddexp
    for t in xrange(1, n_steps):
        for this_l in xrange(n_classes):
            for prev_l in xrange(n_classes):
                a = alpha[t - 1, prev_l]
                c = transitions[prev_l, this_l]
                o = observations[t, this_l]
                # We are accumulating in this, but with log_add_exp instead
                # of just a normal addition (or max in case of Viterbi).
                e = alpha[t, this_l]
                if np.isnan(e):
                    alpha[t, this_l] = a + c + o
                else:
                    alpha[t, this_l] = reducer(e, a + c + o)
    if return_alpha:
        return alpha
    else:
        if viterbi:
            return alpha[-1].max(axis=0)
        else:
            return log_sum_exp(alpha[-1], axis=0)


def forward_np_fast(observations, transitions, viterbi=False,
                    return_alpha=False):
    """
    Takes as input:
        - observations, sequence of shape (n_steps, n_classes)
        - transitions, sequence of shape (n_classes, n_classes)
    Probabilities have to be given in the log space.
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
    n_steps, n_classes = observations.shape
    alpha = np.empty((n_steps, n_classes)) * np.nan
    alpha[0, :] = observations[0:1]
    for t in xrange(1, n_steps):
        a = alpha[t - 1, :, np.newaxis]
        c = transitions
        o = observations[t, np.newaxis, :]
        if viterbi:
            alpha[t] = (a + c + o).max(axis=0)
        else:
            alpha[t] = log_sum_exp(a + c + o, axis=0)
    if return_alpha:
        return alpha
    else:
        if viterbi:
            return alpha[-1].max(axis=0)
        else:
            return log_sum_exp(alpha[-1], axis=0)


def test_forward_np_slow():
    print "Testing slow numpy function..."
    for i in xrange(len(observations_test)):
        print i,
        # No Viterbi
        alpha = forward_np_slow(
            np.log(observations_test[i]),
            np.log(transitions_test[i]),
            viterbi=False,
            return_alpha=True,
        )
        np.testing.assert_allclose(
            np.exp(alpha[-1]),
            alpha_last_test[i],
            rtol=1e-6
        )
        # Viterbi
        logprob = forward_np_slow(
            np.log(observations_test[i]),
            np.log(transitions_test[i]),
            viterbi=True,
            return_alpha=False
        )
        np.testing.assert_allclose(
            np.exp(logprob),
            best_path_prob_test[i],
            rtol=1e-6
        )
    print "OK"


def test_forward_np_fast():
    print "Testing vectorized function..."
    for i in xrange(30):
        print i,
        # Prepare test elements
        seq_length = np.random.randint(1, 20)
        nb_tags = np.random.randint(1, 100)
        obs = np.random.rand(seq_length, nb_tags)
        chain = np.random.rand(nb_tags, nb_tags)
        # No Viterbi
        alpha1 = forward_np_slow(np.log(obs), np.log(chain))
        alpha2 = forward_np_fast(np.log(obs), np.log(chain))
        np.testing.assert_allclose(alpha1, alpha2, rtol=1e-6)
        # Viterbi
        alpha1 = forward_np_slow(np.log(obs), np.log(chain), viterbi=True)
        alpha2 = forward_np_fast(np.log(obs), np.log(chain), viterbi=True)
        np.testing.assert_allclose(alpha1, alpha2, rtol=1e-6)
    print "OK"


def test_forward_theano():
    print "Testing theano function..."
    observations_input_test = theano.tensor.matrix()
    transitions_input_test = theano.tensor.matrix()
    f_theano_no_viterbi = theano.function(
        inputs=[observations_input_test, transitions_input_test],
        outputs=crf.forward(
            observations_input_test,
            transitions_input_test,
            viterbi=False,
            return_alpha=True
        )
    )
    f_theano_viterbi = theano.function(
        inputs=[observations_input_test, transitions_input_test],
        outputs=crf.forward(
            observations_input_test,
            transitions_input_test,
            viterbi=True,
            return_alpha=True
        )
    )
    for i in xrange(30):
        print i,
        seq_length = np.random.randint(2, 20)
        nb_tags = np.random.randint(1, 100)
        obs = np.random.rand(seq_length, nb_tags).astype(np.float32)
        chain = np.random.rand(nb_tags, nb_tags).astype(np.float32)
        # No Viterbi
        alpha1 = forward_np_fast(
            np.log(obs),
            np.log(chain),
            viterbi=False,
            return_alpha=True
        )
        alpha2 = f_theano_no_viterbi(
            np.log(obs),
            np.log(chain)
        )
        np.testing.assert_allclose(alpha1[-1], alpha2[-1], rtol=1e-4)
        # Viterbi
        alpha1 = forward_np_fast(
            np.log(obs),
            np.log(chain),
            viterbi=True,
            return_alpha=True
        )
        alpha2 = f_theano_viterbi(
            np.log(obs),
            np.log(chain)
        )
        np.testing.assert_allclose(alpha1[-1], alpha2[-1], rtol=1e-4)
    print "OK"


assert len(observations_test) == len(transitions_test) == len(alpha_last_test)
assert len(observations_test) == len(best_path_prob_test)

test_forward_np_slow()
test_forward_np_fast()
test_forward_theano()
