import numpy as np
import theano
import theano.tensor as T
from utils import create_shared, random_weights

floatX = theano.config.floatX
device = theano.config.device


class HiddenLayer(object):
    """
    Hidden layer with or without bias.
    Input: tensor of dimension (dim*, input_dim)
    Output: tensor of dimension (dim*, output_dim)
    """

    def __init__(self, input_dim, output_dim, bias=True, activation='sigmoid',
                 name='hidden_layer'):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.name = name
        if activation is None:
            self.activation = None
        elif activation == 'tanh':
            self.activation = T.tanh
        elif activation == 'sigmoid':
            self.activation = T.nnet.sigmoid
        elif activation == 'softmax':
            self.activation = T.nnet.softmax
        else:
            raise Exception("Unknown activation function: " % activation)

        # Initialize weights and bias
        self.weights = create_shared(
            random_weights((input_dim, output_dim)),
            name + '__weights'
        )

        self.bias = create_shared(np.zeros((output_dim,)), name + '__bias')

        # Define parameters
        if self.bias:
            self.params = [self.weights, self.bias]
        else:
            self.params = [self.weights]

    def link(self, input):
        """
        The input has to be a tensor with the right
        most dimension equal to input_dim.
        """
        self.input = input
        self.linear_output = T.dot(self.input, self.weights)
        if self.bias:
            self.linear_output = self.linear_output + self.bias
        if self.activation is None:
            self.output = self.linear_output
        else:
            self.output = self.activation(self.linear_output)
        return self.output


class EmbeddingLayer(object):
    """
    Embedding layer: word embeddings representations
    Input: tensor of dimension (dim*) with values in range(0, input_dim)
    Output: tensor of dimension (dim*, output_dim)
    """

    def __init__(self, input_dim, output_dim, name='embedding_layer'):
        """
        Typically, input_dim is the vocabulary size,
        and output_dim the embedding dimension.
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name

        # Randomly generate weights
        self.embeddings = create_shared(
            random_weights((input_dim, output_dim)),
            self.name + '__embeddings'
        )

        # Define parameters
        self.params = [self.embeddings]

    def link(self, input):
        """
        Return the embeddings of the given indexes.
        Input: tensor of shape (batch_size, sentence_length)
        Output: tensor of shape (batch_size, sentence_length, output_dim)
        """
        self.input = input
        # concat_indexes = self.input.flatten()
        #  __TODO__:check that
        # if device == 'cpu':
        #     indexed_rows = theano.sparse_grad(
        #         self.weights[concatenated_input]
        #     )
        # else:
        self.output = self.embeddings[self.input]
        return self.output
