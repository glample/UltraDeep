import numpy as np
import theano
import theano.tensor as T
from utils import create_shared, random_weights
from layer import DropoutLayer

floatX = theano.config.floatX
device = theano.config.device


class RNN(object):
    """
    Recurrent neural network. Can be used with or without batches.
    Without batches:
        Input: matrix of dimension (sequence_length, input_dim)
        Output: vector of dimension (output_dim)
    With batches:
        Input: tensor3 of dimension (sequence_length, batch_size, input_dim)
        Output: matrix of dimension (batch_size, output_dim)
    """

    def __init__(self, input_dim, hidden_dim, activation=T.nnet.sigmoid,
                 with_batch=True, name='RNN'):
        """
        Initialize neural network.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.with_batch = with_batch
        self.name = name

        # Randomly generate weights
        self.w_x = create_shared(random_weights((input_dim, hidden_dim)), name + '__w_x')
        self.w_h = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_h')

        # Initialize the bias vector and h_0 to zero vectors
        self.b_h = create_shared(np.zeros((hidden_dim,)), name + '__b_h')
        self.h_0 = create_shared(np.zeros((hidden_dim,)), name + '__h_0')

        # Define parameters
        self.params = [self.w_x, self.w_h, self.b_h, self.h_0]

    def link(self, input):
        """
        Propagate the input through the network and return the last hidden vector.
        The whole sequence is also accessible through self.h
        """

        def recurrence(x_t, h_tm1):
            return self.activation(T.dot(x_t, self.w_x) + T.dot(h_tm1, self.w_h) + self.b_h)

        # If we used batches, we have to permute the first and second dimension.
        if self.with_batch:
            self.input = input.dimshuffle(1, 0, 2)
            outputs_info = T.alloc(self.h_0, self.input.shape[1], self.hidden_dim)
        else:
            self.input = input
            outputs_info = self.h_0

        h, _ = theano.scan(
            fn=recurrence,
            sequences=self.input,
            outputs_info=outputs_info,
            n_steps=self.input.shape[0]
        )
        self.h = h
        self.output = h[-1]

        return self.output


class LSTM(object):
    """
    Long short-term memory (LSTM). Can be used with or without batches.
    Without batches:
        Input: matrix of dimension (sequence_length, input_dim)
        Output: vector of dimension (output_dim)
    With batches:
        Input: tensor3 of dimension (sequence_length, batch_size, input_dim)
        Output: matrix of dimension (batch_size, output_dim)
    """

    def __init__(self, input_dim, hidden_dim, with_batch=True, name='LSTM'):
        """
        Initialize neural network.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.with_batch = with_batch
        self.name = name

        # Input gate weights
        self.w_xi = create_shared(random_weights((input_dim, hidden_dim)), name + '__w_xi')
        self.w_hi = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_hi')
        self.w_ci = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_ci')

        # Forget gate weights
        # self.w_xf = create_shared(random_weights((input_dim, hidden_dim)), name + '__w_xf')
        # self.w_hf = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_hf')
        # self.w_cf = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_cf')

        # Output gate weights
        self.w_xo = create_shared(random_weights((input_dim, hidden_dim)), name + '__w_xo')
        self.w_ho = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_ho')
        self.w_co = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_co')

        # Cell weights
        self.w_xc = create_shared(random_weights((input_dim, hidden_dim)), name + '__w_xc')
        self.w_hc = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_hc')

        # Initialize the bias vectors, c_0 and h_0 to zero vectors
        self.b_i = create_shared(np.zeros((hidden_dim,)), name + '__b_i')
        self.b_f = create_shared(np.zeros((hidden_dim,)), name + '__b_f')
        self.b_c = create_shared(np.zeros((hidden_dim,)), name + '__b_c')
        self.b_o = create_shared(np.zeros((hidden_dim,)), name + '__b_o')
        self.c_0 = create_shared(np.zeros((hidden_dim,)), name + '__c_0')
        self.h_0 = create_shared(np.zeros((hidden_dim,)), name + '__h_0')

        # Define parameters
        self.params = [self.w_xi, self.w_hi, self.w_ci,
                       # self.w_xf, self.w_hf, self.w_cf,
                       self.w_xo, self.w_ho, self.w_co,
                       self.w_xc, self.w_hc,
                       self.b_i, self.b_c, self.b_o,  # self.b_f, 
                       self.c_0, self.h_0]

    def link(self, input):
        """
        Propagate the input through the network and return the last hidden vector.
        The whole sequence is also accessible through self.h
        """

        def recurrence(x_t, c_tm1, h_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.w_xi) + T.dot(h_tm1, self.w_hi) + T.dot(c_tm1, self.w_ci) + self.b_i)
            # f_t = T.nnet.sigmoid(T.dot(x_t, self.w_xf) + T.dot(h_tm1, self.w_hf) + T.dot(c_tm1, self.w_cf) + self.b_f)
            c_t = (1 - i_t) * c_tm1 + i_t * T.tanh(T.dot(x_t, self.w_xc) + T.dot(h_tm1, self.w_hc) + self.b_c)
            o_t = T.nnet.sigmoid(T.dot(x_t, self.w_xo) + T.dot(h_tm1, self.w_ho) + T.dot(c_t, self.w_co) + self.b_o)
            h_t = o_t * T.tanh(c_t)
            return [c_t, h_t]

        # If we used batches, we have to permute the first and second dimension.
        if self.with_batch:
            self.input = input.dimshuffle(1, 0, 2)
            outputs_info = [T.alloc(x, self.input.shape[1], self.hidden_dim) for x in [self.c_0, self.h_0]]
        else:
            self.input = input
            outputs_info = [self.c_0, self.h_0]

        [_, h], _ = theano.scan(
            fn=recurrence,
            sequences=self.input,
            outputs_info=outputs_info,
            n_steps=self.input.shape[0]
        )
        self.h = h
        self.output = h[-1]

        return self.output


class GRU(object):
    """
    Gated Recurrent Units (GRU). Can be used with or without batches.
    Without batches:
        Input: matrix of dimension (sequence_length, input_dim)
        Output: vector of dimension (output_dim)
    With batches:
        Input: tensor3 of dimension (sequence_length, batch_size, input_dim)
        Output: matrix of dimension (batch_size, output_dim)
    """

    def __init__(self, input_dim, hidden_dim, with_batch=True, name='LSTM'):
        """
        Initialize neural network.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.with_batch = with_batch
        self.name = name

        # Update gate weights and bias
        self.w_z = create_shared(random_weights((input_dim, hidden_dim)), name + '__w_z')
        self.u_z = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__u_z')
        self.b_z = create_shared(np.zeros((hidden_dim,)), name + '__b_z')

        # Reset gate weights and bias
        self.w_r = create_shared(random_weights((input_dim, hidden_dim)), name + '__w_r')
        self.u_r = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__u_r')
        self.b_r = create_shared(np.zeros((hidden_dim,)), name + '__b_r')

        # New memory content weights and bias
        self.w_c = create_shared(random_weights((input_dim, hidden_dim)), name + '__w_c')
        self.u_c = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__u_c')
        self.b_c = create_shared(np.zeros((hidden_dim,)), name + '__b_c')

        # Initialize the bias vector, h_0, to the zero vector
        self.h_0 = create_shared(np.zeros((hidden_dim,)), name + '__h_0')

        # Define parameters
        self.params = [self.w_z, self.u_z, self.b_z,
                       self.w_r, self.u_r, self.b_r,
                       self.w_c, self.u_c, self.b_c,
                       self.h_0]

    def link(self, input):
        """
        Propagate the input through the network and return the last hidden vector.
        The whole sequence is also accessible through self.h
        """

        def recurrence(x_t, h_tm1):
            z_t = T.nnet.sigmoid(T.dot(x_t, self.w_z) + T.dot(h_tm1, self.u_z) + self.b_z)
            r_t = T.nnet.sigmoid(T.dot(x_t, self.w_r) + T.dot(h_tm1, self.u_r) + self.b_r)
            c_t = T.tanh(T.dot(x_t, self.w_c) + T.dot(r_t * h_tm1, self.u_c) + self.b_c)
            h_t = (1 - z_t) * h_tm1 + z_t * c_t
            return h_t

        # If we used batches, we have to permute the first and second dimension.
        if self.with_batch:
            self.input = input.dimshuffle(1, 0, 2)
            outputs_info = T.alloc(self.h_0, self.input.shape[1], self.hidden_dim)
        else:
            self.input = input
            outputs_info = self.h_0

        h, _ = theano.scan(
            fn=recurrence,
            sequences=self.input,
            outputs_info=outputs_info,
            n_steps=self.input.shape[0]
        )
        self.h = h
        self.output = h[-1]

        return self.output


class FLSTM(object):
    """
    Recurrent neural network with feedback.
    Can be used with or without batches.
    Without batches:
        Input: matrix of dimension (sequence_length, input_dim)
        Output:
            - h: matrix of dimension (sequence_length, hidden_dim)
            - s: matrix of dimension (sequence_length, output_dim)
            - y: vector of dimension (sequence_length,)
    With batches:
        Input: tensor3 of dimension (sequence_length, batch_size, input_dim)
        Output:
            - h: tensor3 of dimension (batch_size, sequence_length, hidden_dim)
            - s: tensor3 of dimension (batch_size, sequence_length, output_dim)
            - y: matrix of dimension (batch_size, sequence_length)
    """

    def __init__(self, input_dim, hidden_dim, output_emb_dim, output_dim,
                 with_batch=True, name='LSTM'):
        """
        Initialize neural network.
          - input_dim: dimension of input vectors
          - hidden_dim: dimension of hidden vectors
          - output_emb_dim: dimension of output embeddings
          - output_dim: number of possible outputs
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_emb_dim = output_emb_dim
        self.output_dim = output_dim
        self.with_batch = with_batch
        self.name = name

        # Input gate weights
        self.w_xi = create_shared(random_weights((input_dim, hidden_dim)), name + '__w_xi')
        self.w_hi = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_hi')
        self.w_yi = create_shared(random_weights((output_emb_dim, hidden_dim)), name + '__w_yi')
        self.w_ci = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_ci')

        # Forget gate weights
        self.w_xf = create_shared(random_weights((input_dim, hidden_dim)), name + '__w_xf')
        self.w_hf = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_hf')
        self.w_yf = create_shared(random_weights((output_emb_dim, hidden_dim)), name + '__w_yf')
        self.w_cf = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_cf')

        # Output gate weights
        self.w_xo = create_shared(random_weights((input_dim, hidden_dim)), name + '__w_xo')
        self.w_ho = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_ho')
        self.w_yo = create_shared(random_weights((output_emb_dim, hidden_dim)), name + '__w_yo')
        self.w_co = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_co')

        # Cell weights
        self.w_xc = create_shared(random_weights((input_dim, hidden_dim)), name + '__w_xc')
        self.w_hc = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_hc')
        self.w_yc = create_shared(random_weights((output_emb_dim, hidden_dim)), name + '__w_yc')

        # Initialize the bias vectors, c_0 and h_0 to zero vectors
        self.b_i = create_shared(np.zeros((hidden_dim,)), name + '__b_i')
        self.b_f = create_shared(np.zeros((hidden_dim,)), name + '__b_f')
        self.b_c = create_shared(np.zeros((hidden_dim,)), name + '__b_c')
        self.b_o = create_shared(np.zeros((hidden_dim,)), name + '__b_o')
        self.c_0 = create_shared(np.zeros((hidden_dim,)), name + '__c_0')
        self.h_0 = create_shared(np.zeros((hidden_dim,)), name + '__h_0')
        # self.y_0 = create_shared(np.zeros((output_emb_dim,)), name + '__y_0')

        # Weights for projection to final output, and outputs embeddings
        self.embeddings = create_shared(random_weights((output_dim + 1, output_emb_dim)), name + '__embeddings')
        self.weights = create_shared(random_weights((hidden_dim, output_dim)), name + '__weights')
        self.bias = create_shared(random_weights((output_dim,)), name + '__bias')

        # Define parameters
        self.params = [self.w_xi, self.w_hi, self.w_yi, self.w_ci,
                       self.w_xf, self.w_hf, self.w_yf, self.w_cf,
                       self.w_xo, self.w_ho, self.w_yo, self.w_co,
                       self.w_xc, self.w_hc, self.w_yc,
                       self.b_i, self.b_c, self.b_o, self.b_f,
                       self.c_0, self.h_0,  # self.y_0,
                       self.embeddings, self.weights, self.bias]

    def link(self, input):
        """
        Propagate the input through the network and return the last hidden vector.
        The whole sequence is also accessible through self.h
        """

        def recurrence(x_t, c_tm1, h_tm1, _, y_tm1, embeddings, weights, bias):
            y_tm1_emb = embeddings[y_tm1]
            i_t = T.nnet.sigmoid(T.dot(x_t, self.w_xi) + T.dot(h_tm1, self.w_hi) + T.dot(y_tm1_emb, self.w_yi) + T.dot(c_tm1, self.w_ci) + self.b_i)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.w_xf) + T.dot(h_tm1, self.w_hf) + T.dot(y_tm1_emb, self.w_yf) + T.dot(c_tm1, self.w_cf) + self.b_f)
            c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.w_xc) + T.dot(h_tm1, self.w_hc) + T.dot(y_tm1_emb, self.w_yc) + self.b_c)
            o_t = T.nnet.sigmoid(T.dot(x_t, self.w_xo) + T.dot(h_tm1, self.w_ho) + T.dot(y_tm1_emb, self.w_yo) + T.dot(c_t, self.w_co) + self.b_o)
            h_t = o_t * T.tanh(c_t)
            if self.with_batch:
                s_t = T.nnet.softmax(T.dot(h_t, weights) + bias)
            else:
                s_t = T.flatten(T.nnet.softmax(T.dot(h_t, weights) + bias), 1)
            y_t = T.cast(T.argmax(s_t, axis=-1), 'int32')
            return [c_t, h_t, s_t, y_t]

        # If we used batches, we have to permute the first and second dimension.
        if self.with_batch:
            self.input = input.dimshuffle(1, 0, 2)
            outputs_info = [
                T.alloc(self.c_0, self.input.shape[1], self.hidden_dim),
                T.alloc(self.h_0, self.input.shape[1], self.hidden_dim),
                T.alloc(np.zeros(self.output_dim).astype(np.float32), self.input.shape[1], self.output_dim),
                T.alloc(T.cast(self.output_dim, 'int32'), self.input.shape[1])
            ]
        else:
            self.input = input
            outputs_info = [
                self.c_0,
                self.h_0,
                np.zeros(self.output_dim).astype(np.float32),
                T.alloc(T.cast(self.output_dim, 'int32'))
            ]

        [_, h, s, y], _ = theano.scan(
            fn=recurrence,
            sequences=self.input,
            outputs_info=outputs_info,
            n_steps=self.input.shape[0],
            non_sequences=[self.embeddings, self.weights, self.bias]
        )
        self.h = h
        self.s = s
        self.y = y


class DeepLSTM(object):
    """
    Deep LSTM. Can be used with or without batches.
    Can also be used with dropout.
    Without batches:
        Input: matrix of dimension (sequence_length, input_dim)
        Output: vector of dimension (output_dim)
    With batches:
        Input: tensor3 of dimension (sequence_length, batch_size, input_dim)
        Output: matrix of dimension (batch_size, output_dim)
    """

    def __init__(self, input_dim, hidden_dim, dropout=None,
                 with_batch=True, name='DeepLSTM'):
        """
        Initialize neural network.
        """
        if type(hidden_dim) is int:
            hidden_dim = [hidden_dim]
        assert type(hidden_dim) is list and len(hidden_dim) >= 1
        assert dropout is None or type(dropout) is float
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.with_batch = with_batch
        self.name = name

        # Create all LSTMs
        input_dims = [input_dim] + hidden_dim[:-1]
        self.lstms = [
            LSTM(a, b, with_batch=with_batch, name='%s_%i' % (name, i))
            for i, (a, b) in enumerate(zip(input_dims, hidden_dim))
        ]

        # Create dropout layers
        if dropout is not None:
            self.dropout_layers = [
                DropoutLayer(p=dropout)
                for _ in xrange(len(hidden_dim))
            ]

        # Define parameters
        self.params = sum([lstm.params for lstm in self.lstms], [])

    def link(self, input, is_train=None):
        """
        Propagate the input through the network.
        """
        assert not ((is_train is None) ^ (self.dropout is None))
        self.layer_outputs = []

        for i in xrange(len(self.lstms)):
            # Apply dropout
            if self.dropout is not None:
                layer_input = T.switch(
                    T.neq(is_train, 0),
                    self.dropout_layers[i].link(input),
                    self.dropout * input
                )
            else:
                layer_input = input
            self.lstms[i].link(layer_input)
            self.layer_outputs.append(self.lstms[i].h)
            input = self.lstms[i].h

        self.output = input
        # self.layer_outputs = T.stack(*self.layer_outputs)

        return self.output
