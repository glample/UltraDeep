import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX
device = theano.config.device


def get_drange(shape, activation=T.nnet.sigmoid):
    """
    Get a domain of distribution to randomly initialize weight matrices.
    """
    drange = np.sqrt(6. / (np.sum(shape)))
    if activation == T.nnet.sigmoid:
        drange *= 4
    return drange


def random_weights(shape):
    """
    Return a matrix of a given shape, with weights randomly
    distributed in the interval [-1, 1].
    """
    return np.random.uniform(low=-1.0, high=1.0, size=shape)


def create_shared(value, name):
    """
    Create a shared object of a numpy array.
    """
    return theano.shared(value=np.array(value, dtype=floatX), name=name)
