import numpy as np
import theano
import theano.tensor as T
import os
import cPickle
from utils import *

floatX = theano.config.floatX
device = theano.config.device


class Structure(object):
    """
    Network structure. For easy saving and loading.
    """
    def __init__(self, name, dump_path='/home/guillaume/workspace/saved_models'):
        self.name = name
        self.dump_path = os.path.join(dump_path, name)
        self.components = {}
        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)

    def add_component(self, component):
        if component.name in self.components:
            raise Exception("%s is already a component of this network!" % component.name)
        self.components[component.name] = component

    def dump(self, message):
        # Write components
        for name, component in self.components.items():
            for param in component.params:
                param_path = os.path.join(self.dump_path, "%s__%s.pkl" % (name, param.name))
                cPickle.dump(param.get_value(), open(param_path, 'w'))
        # Write information
        messages_path = os.path.join(self.dump_path, "messages.txt")
        with open(messages_path, "a") as f:
            f.write(message + "\n")
        params_shapes_path = os.path.join(self.dump_path, "params_shapes.txt")
        with open(params_shapes_path, "w") as f:
            for name, component in self.components.items():
                for param in component.params:
                    f.write("%s__%s : %s\n" % (name, param.name, param.get_value().shape))

    def load(self):
        # Load components
        for name, component in self.components.items():
            for param in component.params:
                param_path = os.path.join(self.dump_path, "%s__%s.pkl" % (name, param.name))
                param.set_value(cPickle.load(open(param_path, 'r')))


class RNN(object):
    """
    Recurrent neural network. Can be used with or without batches.
    Without batches:
        Input: matrix of dimension (sequence_length, input_dim)
        Output: vector of dimension (output_dim)
    With batches:
        Input: tensor3 of dimension (batch_size, sequence_length, input_dim)
        Output: matrix of dimension (batch_size, output_dim)
    """

    def __init__(self, input_dim, hidden_dim, activation=T.nnet.sigmoid, with_batch=True, name='RNN'):
        """
        Initialize neural network.
        The output layer is not defined here, and has to be added on the output of the RNN.
        Indeed, we don't know what is the expected type of output we want to have (MSE,
        binary, multiclass, etc.)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.with_batch = with_batch
        self.name = name

        # Domain range for weights initialization
        drange_x = get_drange((input_dim, hidden_dim), activation)
        drange_h = get_drange((hidden_dim, hidden_dim), activation)

        # Randomly generate weights
        self.w_x = create_shared(drange_x * random_weights((input_dim, hidden_dim)), name + '_w_x')
        self.w_h = create_shared(drange_h * random_weights((hidden_dim, hidden_dim)), name + '_w_h')

        # Initialize the bias vector and h_0 to zero vectors
        self.b_h = create_shared(np.zeros((hidden_dim,)), name + '_b_h')
        self.h_0 = create_shared(np.zeros((hidden_dim,)), name + '_h_0')

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
    Recurrent neural network. Can be used with or without batches.
    Without batches:
        Input: matrix of dimension (sequence_length, input_dim)
        Output: vector of dimension (output_dim)
    With batches:
        Input: tensor3 of dimension (batch_size, sequence_length, input_dim)
        Output: matrix of dimension (batch_size, output_dim)
    """

    def __init__(self, input_dim, hidden_dim, with_batch=True, name='LSTM'):
        """
        Initialize neural network.
        The output layer is not defined here, and has to be added on the output of the RNN.
        Indeed, we don't know what is the expected type of output we want to have (MSE,
        binary, multiclass, etc.)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.with_batch = with_batch
        self.name = name

        # Domain range for weights initialization
        drange_x = get_drange((input_dim, hidden_dim))
        drange_h = get_drange((hidden_dim, hidden_dim))

        # Input gate weights
        self.w_xi = create_shared(drange_x * random_weights((input_dim, hidden_dim)), name + '_w_xi')
        self.w_hi = create_shared(drange_h * random_weights((hidden_dim, hidden_dim)), name + '_w_hi')
        self.w_ci = create_shared(drange_h * random_weights((hidden_dim, hidden_dim)), name + '_w_ci')

        # Forget gate weights
        self.w_xf = create_shared(drange_x * random_weights((input_dim, hidden_dim)), name + '_w_xf')
        self.w_hf = create_shared(drange_h * random_weights((hidden_dim, hidden_dim)), name + '_w_hf')
        self.w_cf = create_shared(drange_h * random_weights((hidden_dim, hidden_dim)), name + '_w_cf')

        # Output gate weights
        self.w_xo = create_shared(drange_x * random_weights((input_dim, hidden_dim)), name + '_w_xo')
        self.w_ho = create_shared(drange_h * random_weights((hidden_dim, hidden_dim)), name + '_w_ho')
        self.w_co = create_shared(drange_h * random_weights((hidden_dim, hidden_dim)), name + '_w_co')

        # Cell weights
        self.w_xc = create_shared(drange_x * random_weights((input_dim, hidden_dim)), name + '_w_xc')
        self.w_hc = create_shared(drange_h * random_weights((hidden_dim, hidden_dim)), name + '_w_hc')

        # Initialize the bias vectors, c_0 and h_0 to zero vectors
        self.b_i = create_shared(np.zeros((hidden_dim,)), name + '_b_i')
        self.b_f = create_shared(np.zeros((hidden_dim,)), name + '_b_f')
        self.b_c = create_shared(np.zeros((hidden_dim,)), name + '_b_c')
        self.b_o = create_shared(np.zeros((hidden_dim,)), name + '_b_o')
        self.c_0 = create_shared(np.zeros((hidden_dim,)), name + '_c_0')
        self.h_0 = create_shared(np.zeros((hidden_dim,)), name + '_h_0')

        # Define parameters
        self.params = [self.w_xi, self.w_hi, self.w_ci,
                       self.w_xf, self.w_hf, self.w_cf,
                       self.w_xo, self.w_ho, self.w_co,
                       self.w_xc, self.w_hc,
                       self.b_i, self.b_f, self.b_c, self.b_o,
                       self.c_0, self.h_0]

    def link(self, input):
        """
        Propagate the input through the network and return the last hidden vector.
        The whole sequence is also accessible through self.h
        """

        def recurrence(x_t, c_tm1, h_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.w_xi) + T.dot(h_tm1, self.w_hi) + T.dot(c_tm1, self.w_ci) + self.b_i)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.w_xf) + T.dot(h_tm1, self.w_hf) + T.dot(c_tm1, self.w_cf) + self.b_f)
            c_t = f_t * c_tm1 + i_t * T.nnet.sigmoid(T.dot(x_t, self.w_xc) + T.dot(h_tm1, self.w_hc) + self.b_c)
            o_t = T.nnet.sigmoid(T.dot(x_t, self.w_xo) + T.dot(h_tm1, self.w_ho) + T.dot(c_t, self.w_co) + self.b_o)
            h_t = o_t * T.nnet.sigmoid(c_t)
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
