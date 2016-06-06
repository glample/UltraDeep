import numpy as np
import theano
import theano.tensor as T
from utils import create_shared, random_weights
from theano.tensor.nnet import conv
import pooling


floatX = theano.config.floatX
device = theano.config.device


class Conv1DLayer(object):

    def __init__(self, nb_filters, stack_size, filter_height, wide, emb_dim, name):
        """
        1D convolutional layer: 1D Row-wise convolution.
        Requires to know the dimension of the embeddings.
        """
        self.nb_filters = nb_filters
        self.stack_size = stack_size
        self.filter_height = filter_height
        self.wide = wide
        self.emb_dim = emb_dim
        self.filter_shape = (emb_dim, nb_filters, stack_size, filter_height, 1)

        # _TODO_ check initialization
        # fan_in = in_fmaps * 1 * width
        # fan_out = out_fmaps * 1 * width
        # W_bound = numpy.sqrt(6./(fan_in+fan_out))
        filters_values = np.asarray(
            np.random.normal(0, 0.05, size=self.filter_shape),
            dtype=theano.config.floatX
        )
        self.filters = create_shared(filters_values, name + '__filters')
        self.bias = create_shared(np.zeros((nb_filters, emb_dim)), name + '__bias')

        # parameters in the layer
        self.params = [self.filters, self.bias]

    def link(self, input):
        self.input = input
        conv_list = []
        for i in range(self.emb_dim):
            conv_out = conv.conv2d(
                input=self.input[:, :, :, i:i + 1],
                filters=self.filters[i],
                border_mode=('full' if self.wide else 'valid')
            )
            conv_list.append(conv_out)

        self.conv_out = T.concatenate(conv_list, axis=3)

        # bias + squash function
        self.linear_output = self.conv_out + self.bias.dimshuffle('x', 0, 'x', 1)
        self.output = T.tanh(self.linear_output)

        return self.output


class Conv2DLayerOld(object):
    """
    2D Convolutional neural layer.
    """

    def __init__(self, nb_filters, stack_size, filter_height, filter_width, wide, name):
        """
        Construct a convolutional layer
        `wide`:
            False: only apply filter to complete patches of the image.
            Generates output of shape: image_shape - filter_shape + 1
            True: zero-pads image to multiple of filter shape to generate
            output of shape: image_shape + filter_shape - 1
        """
        self.nb_filters = nb_filters
        self.stack_size = stack_size
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.wide = wide
        self.name = name
        self.filter_shape = (nb_filters, stack_size, filter_height, filter_width)

        fan_in = stack_size * filter_height * filter_width       # number of inputs to each hidden unit
        fan_out = ((nb_filters * filter_height * filter_width))  # each unit in the lower layer receives a gradient from
        drange = np.sqrt(6. / (fan_in + fan_out))                # initialize filters with random values

        self.filters = create_shared(drange * random_weights(self.filter_shape), name + '__filters')
        self.bias = create_shared(np.zeros((nb_filters,)), name + '__bias')

        # parameters in the layer
        self.params = [self.filters, self.bias]

    def link(self, input):
        """
        Convolve input feature maps with filters.
        Input: Feature map of dimension (batch_size, stack_size, nb_rows, nb_cols)
        Output: Feature map of dimension (batch_size, nb_filters, output_rows, output_cols)
        """
        self.input = input

        # convolutional layer
        self.conv_out = conv.conv2d(
            input=self.input,
            filters=self.filters,
            border_mode=('full' if self.wide else 'valid'),
            filter_shape=self.filter_shape
        )

        # bias + squash function
        self.linear_output = self.conv_out + self.bias.dimshuffle('x', 0, 'x', 'x')
        self.output = T.tanh(self.linear_output)

        return self.output


class Conv2DLayer(object):
    """
    2D Convolutional neural layer.
    """

    def __init__(self, nb_filters, stack_size, filter_height, filter_width, border_mode, stride, name):
        """
        Construct a convolutional layer.
        """
        self.nb_filters = nb_filters
        self.stack_size = stack_size
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.border_mode = border_mode
        self.filter_shape = (nb_filters, stack_size, filter_height, filter_width)
        self.stride = stride
        self.name = name

        fan_in = stack_size * filter_height * filter_width       # number of inputs to each hidden unit
        fan_out = ((nb_filters * filter_height * filter_width))  # each unit in the lower layer receives a gradient from
        drange = np.sqrt(6. / (fan_in + fan_out))                # initialize filters with random values

        self.filters = create_shared(drange * random_weights(self.filter_shape), name + '__filters')
        self.bias = create_shared(np.ones((nb_filters,)), name + '__bias')

        # parameters in the layer
        self.params = [self.filters, self.bias]

    def link(self, input):
        """
        Convolve input feature maps with filters.
        Input: Feature map of dimension (batch_size, stack_size, nb_rows, nb_cols)
        Output: Feature map of dimension (batch_size, nb_filters, output_rows, output_cols)
        """

        # convolutional layer
        self.conv_out = T.nnet.conv2d(
            input=input,
            filters=self.filters,
            # input_shape=None, _TODO_ might be faster
            filter_shape=self.filter_shape,
            border_mode=self.border_mode,
            subsample=self.stride,
            filter_flip=False,
            # image_shape=None
        )

        # bias + squash function
        self.linear_output = self.conv_out + self.bias.dimshuffle('x', 0, 'x', 'x')
        self.output = T.nnet.relu(self.linear_output)

        return self.output


class Conv1DLayerKMaxPooling(object):
    """
    1D Convolutional neural layer with k-max pooling.
    """

    def __init__(self, nb_filters, stack_size, filter_height, wide, emb_dim, name):
        """
        Construct a convolutional layer
        `wide`:
            False: only apply filter to complete patches of the image.
            Generates output of shape: image_shape - filter_shape + 1
            True: zero-pads image to multiple of filter shape to generate
            output of shape: image_shape + filter_shape - 1
        """
        self.nb_filters = nb_filters
        self.stack_size = stack_size
        self.filter_height = filter_height
        self.wide = wide
        self.emb_dim = emb_dim
        self.name = name
        self.k_max = None

        self.conv1d_layer = Conv1DLayer(
            nb_filters,
            stack_size,
            filter_height,
            wide,
            emb_dim,
            name + "__conv1d_layer"
        )

        # parameters in the layer
        self.params = [self.conv1d_layer.filters, self.conv1d_layer.bias]

    def link(self, input):
        """
        Convolve input feature maps with filters.
        Input: Feature map of dimension (batch_size, stack_size, nb_rows, nb_cols)
        Output: Feature map of dimension (batch_size, nb_filters, output_rows, output_cols)
        """
        if self.k_max is None:
            raise Exception("k_max has not been defined in the layer %s!" % self.name)

        self.input = input

        # 1D convolutional layer
        self.conv1d_layer.link(self.input)
        self.conv_out = self.conv1d_layer.conv_out

        # k-max pooling
        k_max_layer = pooling.KMaxPoolingLayer1(self.k_max)
        self.pooled_out = k_max_layer.link(self.conv_out)

        # bias + squash function
        self.linear_output = self.pooled_out + self.conv1d_layer.bias.dimshuffle('x', 0, 'x', 1)
        self.output = T.tanh(self.linear_output)

        return self.output


class Conv2DLayerKMaxPooling(object):
    """
    2D Convolutional neural layer with k-max pooling.
    """

    def __init__(self, nb_filters, stack_size, filter_height, filter_width, wide, name):
        """
        Construct a convolutional layer
        `wide`:
            False: only apply filter to complete patches of the image.
            Generates output of shape: image_shape - filter_shape + 1
            True: zero-pads image to multiple of filter shape to generate
            output of shape: image_shape + filter_shape - 1
        """
        self.nb_filters = nb_filters
        self.stack_size = stack_size
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.wide = wide
        self.name = name
        self.k_max = None

        self.conv2d_layer = Conv2DLayer(
            nb_filters,
            stack_size,
            filter_height,
            filter_width,
            wide,
            name + "__conv2d_layer"
        )

        # parameters in the layer
        self.params = [self.conv2d_layer.filters, self.conv2d_layer.bias]

    def link(self, input):
        """
        Convolve input feature maps with filters.
        Input: Feature map of dimension (batch_size, stack_size, nb_rows, nb_cols)
        Output: Feature map of dimension (batch_size, nb_filters, output_rows, output_cols)
        """
        if self.k_max is None:
            raise Exception("k_max has not been defined in the layer %s!" % self.name)

        self.input = input

        # 2D convolutional layer
        self.conv2d_layer.link(self.input)
        self.conv_out = self.conv2d_layer.conv_out

        # k-max pooling
        k_max_layer = pooling.KMaxPoolingLayer1(self.k_max)
        self.pooled_out = k_max_layer.link(self.conv_out)

        # bias + squash function
        self.linear_output = self.pooled_out + self.conv2d_layer.bias.dimshuffle('x', 0, 'x', 'x')
        self.output = T.tanh(self.linear_output)

        return self.output
