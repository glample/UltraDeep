import numpy as np
import theano
import theano.tensor as T
import timeit
import layer
import network
import convolution
from scipy.special import expit
from scipy.signal import convolve2d
from scipy.signal import convolve
import pooling

floatX = theano.config.floatX
device = theano.config.device


def test_hidden_layer():

    print "Testing hidden layer..."

    for k in xrange(5):
        print "Layer %i..." % k

        # random parameters
        input_dim = np.random.randint(1, 100)
        output_dim = np.random.randint(1, 100)
        bias = np.random.randint(2)
        activation = 'tanh' if np.random.randint(2) else 'sigmoid'

        # hidden layer
        hidden_layer = layer.HiddenLayer(input_dim, output_dim, bias,
                                         activation, 'test')

        for i in xrange(20):
            print "%i" % i,

            # tests for dimension 1, 2, 3 and 4
            if i % 4 == 0:
                input = T.vector('input_test')
                input_value = np.random.rand(input_dim).astype(floatX)
            elif i % 4 == 1:
                input = T.matrix('input_test')
                input_value = np.random.rand(
                    np.random.randint(100),
                    input_dim
                ).astype(floatX)
            elif i % 4 == 2:
                input = T.tensor3('input_test')
                input_value = np.random.rand(
                    np.random.randint(50),
                    np.random.randint(50),
                    input_dim
                ).astype(floatX)
            else:
                input = T.tensor4('input_test')
                input_value = np.random.rand(
                    np.random.randint(20),
                    np.random.randint(20),
                    np.random.randint(20),
                    input_dim
                ).astype(floatX)

            output = hidden_layer.link(input)

            expected_value = np.dot(
                input_value,
                hidden_layer.weights.get_value()
            )
            if bias:
                expected_value += hidden_layer.bias.get_value()
            if activation == 'tanh':
                expected_value = np.tanh(expected_value)
            else:
                expected_value = expit(expected_value)

            assert expected_value.shape == input_value.shape[:-1] + (output_dim,)
            np.testing.assert_array_almost_equal(
                output.eval({input: input_value}),
                expected_value
            )

        print "OK"

    print "All tests ran successfully for Hidden Layer."


def test_embedding_layer():

    print "Testing embedding layer..."

    for k in xrange(1):
        print "Layer %i..." % k

        # random parameters
        input_dim = np.random.randint(1, 100)
        output_dim = np.random.randint(1, 100)

        # embedding layer
        embedding_layer = layer.EmbeddingLayer(input_dim, output_dim, 'test')

        for i in xrange(40):
            print "%i" % i,

            # tests for dimension 1, 2, 3 and 4
            if i % 4 == 0:
                input = T.ivector('input_test')
                input_value = np.random.randint(
                    low=0,
                    high=input_dim,
                    size=(np.random.randint(low=1, high=50),)
                ).astype(np.int32)
            elif i % 4 == 1:
                input = T.imatrix('input_test')
                input_value = np.random.randint(
                    low=0,
                    high=input_dim,
                    size=(np.random.randint(low=1, high=40),
                          np.random.randint(low=1, high=40))
                ).astype(np.int32)
            elif i % 4 == 2:
                input = T.itensor3('input_test')
                input_value = np.random.randint(
                    low=0,
                    high=input_dim,
                    size=(np.random.randint(low=1, high=30),
                          np.random.randint(low=1, high=30),
                          np.random.randint(low=1, high=30))
                ).astype(np.int32)
            else:
                input = T.itensor4('input_test')
                input_value = np.random.randint(
                    low=0,
                    high=input_dim,
                    size=(np.random.randint(low=1, high=20),
                          np.random.randint(low=1, high=20),
                          np.random.randint(low=1, high=20),
                          np.random.randint(low=1, high=20))
                ).astype(np.int32)

            output = embedding_layer.link(input)

            expected_value = embedding_layer.embeddings.get_value()[input_value]

            assert expected_value.shape == input_value.shape + (output_dim,)
            np.testing.assert_array_almost_equal(
                output.eval({input: input_value}),
                expected_value
            )

        print "OK"

    print "All tests ran successfully for Embedding Layer."


def test_rnn():

    print "Testing RNN without minibatch..."

    # without minibatch
    for k in xrange(5):
        print "Network %i..." % k

        # random parameters
        input_dim = np.random.randint(1, 30)
        hidden_dim = np.random.randint(1, 30)
        activation = T.tanh if np.random.randint(2) else T.nnet.sigmoid
        with_batch = False

        # rnn
        rnn = network.RNN(input_dim, hidden_dim, activation, with_batch, 'rnn')

        for i in xrange(10):
            print "%i" % i,

            input = T.matrix('input_test')
            input_value = np.random.rand(
                np.random.randint(low=1, high=30),
                input_dim
            ).astype(floatX)
            output = rnn.link(input)

            h_t = rnn.h_0.get_value()
            for i in xrange(input_value.shape[0]):
                h_t = np.dot(
                    input_value[i],
                    rnn.w_x.get_value()
                ) + np.dot(h_t, rnn.w_h.get_value()) + rnn.b_h.get_value()
                if activation == T.tanh:
                    h_t = np.tanh(h_t)
                else:
                    h_t = expit(h_t)

            assert h_t.shape == (hidden_dim,)
            np.testing.assert_array_almost_equal(
                output.eval({input: input_value}),
                h_t
            )

        print "OK"

    print "Testing RNN with minibatch..."

    # with minibatch
    for k in xrange(5):
        print "Network %i..." % k

        # random parameters
        input_dim = np.random.randint(1, 30)
        hidden_dim = np.random.randint(1, 30)
        activation = T.tanh if np.random.randint(2) else T.nnet.sigmoid
        with_batch = True

        # hidden layer
        rnn = network.RNN(input_dim, hidden_dim, activation, with_batch, 'RNN')

        for i in xrange(10):
            print "%i" % i,

            input = T.tensor3('input_test')
            input_value = np.random.rand(
                np.random.randint(low=1, high=10),
                np.random.randint(low=1, high=30),
                input_dim
            ).astype(floatX)
            input_value_dimshuffled = np.transpose(input_value, (1, 0, 2))
            output = rnn.link(input)

            h_t = np.array([rnn.h_0.get_value()] * input_value_dimshuffled.shape[1])
            for i in xrange(input_value_dimshuffled.shape[0]):
                h_t = np.dot(
                    input_value_dimshuffled[i],
                    rnn.w_x.get_value()
                ) + np.dot(h_t, rnn.w_h.get_value()) + rnn.b_h.get_value()
                if activation == T.tanh:
                    h_t = np.tanh(h_t)
                else:
                    h_t = expit(h_t)

            assert h_t.shape == (input_value.shape[0], hidden_dim)
            np.testing.assert_array_almost_equal(
                output.eval({input: input_value}),
                h_t
            )

        print "OK"

    print "All tests ran successfully for RNN."


def test_lstm():

    print "Testing LSTM without minibatch..."

    # without minibatch
    for k in xrange(5):
        print "Network %i..." % k

        # random parameters
        input_dim = np.random.randint(1, 30)
        hidden_dim = np.random.randint(1, 30)
        with_batch = False

        # lstm
        lstm = network.LSTM(input_dim, hidden_dim, with_batch, 'LSTM')

        for i in xrange(10):
            print "%i" % i,

            input = T.matrix('input_test')
            input_value = np.random.rand(
                np.random.randint(low=1, high=30),
                input_dim
            ).astype(floatX)
            output = lstm.link(input)

            c_t = lstm.c_0.get_value()
            h_t = lstm.h_0.get_value()
            for i in xrange(input_value.shape[0]):
                x_t = input_value[i]
                i_t = expit(np.dot(x_t, lstm.w_xi.get_value()) + np.dot(h_t, lstm.w_hi.get_value()) + np.dot(c_t, lstm.w_ci.get_value()) + lstm.b_i.get_value())
                f_t = expit(np.dot(x_t, lstm.w_xf.get_value()) + np.dot(h_t, lstm.w_hf.get_value()) + np.dot(c_t, lstm.w_cf.get_value()) + lstm.b_f.get_value())
                c_t = f_t * c_t + i_t * expit(np.dot(x_t, lstm.w_xc.get_value()) + np.dot(h_t, lstm.w_hc.get_value()) + lstm.b_c.get_value())
                o_t = expit(np.dot(x_t, lstm.w_xo.get_value()) + np.dot(h_t, lstm.w_ho.get_value()) + np.dot(c_t, lstm.w_co.get_value()) + lstm.b_o.get_value())
                h_t = o_t * expit(c_t)

            assert h_t.shape == (hidden_dim,)
            np.testing.assert_array_almost_equal(
                output.eval({input: input_value}),
                h_t,
                decimal=3
            )

        print "OK"

    print "Testing LSTM with minibatch..."

    # with minibatch
    for k in xrange(5):
        print "Network %i..." % k

        # random parameters
        input_dim = np.random.randint(1, 30)
        hidden_dim = np.random.randint(1, 30)
        with_batch = True

        # lstm
        lstm = network.LSTM(input_dim, hidden_dim, with_batch, 'LSTM')

        for i in xrange(10):
            print "%i" % i,

            input = T.tensor3('input_test')
            input_value = np.random.rand(
                np.random.randint(low=1, high=10),
                np.random.randint(low=1, high=30),
                input_dim
            ).astype(floatX)
            input_value_dimshuffled = np.transpose(input_value, (1, 0, 2))
            output = lstm.link(input)

            c_t = lstm.c_0.get_value()
            h_t = lstm.h_0.get_value()
            for i in xrange(input_value_dimshuffled.shape[0]):
                x_t = input_value_dimshuffled[i]
                i_t = expit(np.dot(x_t, lstm.w_xi.get_value()) + np.dot(h_t, lstm.w_hi.get_value()) + np.dot(c_t, lstm.w_ci.get_value()) + lstm.b_i.get_value())
                f_t = expit(np.dot(x_t, lstm.w_xf.get_value()) + np.dot(h_t, lstm.w_hf.get_value()) + np.dot(c_t, lstm.w_cf.get_value()) + lstm.b_f.get_value())
                c_t = f_t * c_t + i_t * expit(np.dot(x_t, lstm.w_xc.get_value()) + np.dot(h_t, lstm.w_hc.get_value()) + lstm.b_c.get_value())
                o_t = expit(np.dot(x_t, lstm.w_xo.get_value()) + np.dot(h_t, lstm.w_ho.get_value()) + np.dot(c_t, lstm.w_co.get_value()) + lstm.b_o.get_value())
                h_t = o_t * expit(c_t)

            assert h_t.shape == (input_value.shape[0], hidden_dim)
            np.testing.assert_array_almost_equal(output.eval({input: input_value}), h_t, decimal=3)

        print "OK"

    print "All tests ran successfully for LSTM."


def kmax_pooling(a, k):
    """
    Take as input a 4D array, and return the same array where
    we only take the k biggest elements of the third dimension (3/4).
    """
    ind = np.argsort(a, axis=2)
    sorted_ind = np.sort(ind[:, :, -k:, :], axis=2)
    dim0, dim1, dim2, dim3 = sorted_ind.shape
    indices_dim0 = np.arange(dim0).repeat(dim1 * dim2 * dim3)
    indices_dim1 = np.tile(np.arange(dim1).repeat(dim2 * dim3), dim0)
    # indices_dim2 = np.tile(np.arange(dim2).repeat(dim3), dim0 * dim1)
    indices_dim3 = np.tile(np.arange(dim3), dim0 * dim1 * dim2)
    return a[indices_dim0, indices_dim1, sorted_ind.flatten(), indices_dim3].reshape(sorted_ind.shape)


def test_kmax_pooling_layer_1():
    """
    Doesn't seem to work if k-max is bigger than the third dimension.
    """

    print "Testing k-max Pooling Layer 1..."

    input = T.tensor4('input_test')
    k_max = T.iscalar('')
    kmax_pooling_layer_old = pooling.KMaxPoolingLayer1(k_max)
    output = kmax_pooling_layer_old.link(input)

    for i in xrange(1000):
        if i % 50 == 0:
            print "%i" % i,

        # random parameters
        input_value = np.random.rand(
            np.random.randint(1, 20),
            np.random.randint(1, 20),
            np.random.randint(10, 20),
            np.random.randint(1, 100)
        ).astype(floatX)
        k_max_value = np.random.randint(1, 10)
        expected_value = kmax_pooling(input_value, k_max_value)

        # print k_max_value, input_value.shape, expected_value.shape
        # print output.eval({input:input_value, k_max:k_max_value}).shape

        assert expected_value.shape[:2] + expected_value.shape[3:] == input_value.shape[:2] + input_value.shape[3:]
        assert expected_value.shape[2] in [k_max_value, input_value.shape[2]]
        np.testing.assert_array_almost_equal(output.eval({input: input_value, k_max: k_max_value}), expected_value)

    print "OK"
    print "All tests ran successfully for k-max Pooling Layer 1."


def test_kmax_pooling_layer_2():

    print "Testing k-max Pooling Layer 2..."

    input = T.tensor4('input_test')
    k_max = T.iscalar('')
    kmax_pooling_layer = pooling.KMaxPoolingLayer2(k_max)
    output = kmax_pooling_layer.link(input)

    for i in xrange(1000):
        if i % 50 == 0:
            print "%i" % i,

        # random parameters
        input_value = np.random.rand(
            np.random.randint(1, 20),
            np.random.randint(1, 20),
            np.random.randint(10, 20),
            np.random.randint(1, 100)
        ).astype(floatX)
        k_max_value = np.random.randint(1, 10)
        expected_value = kmax_pooling(input_value, k_max_value)

        # print k_max_value, input_value.shape, expected_value.shape
        # print output.eval({input:input_value, k_max:k_max_value}).shape

        assert expected_value.shape[:2] + expected_value.shape[3:] == input_value.shape[:2] + input_value.shape[3:]
        assert expected_value.shape[2] in [k_max_value, input_value.shape[2]]
        np.testing.assert_array_almost_equal(output.eval({input: input_value, k_max: k_max_value}), expected_value)

    print "OK"
    print "All tests ran successfully for k-max Pooling Layer 2."


def test_conv1d_layer():

    print "Testing 1D convolutional layer..."

    for k in xrange(5):
        print "Layer %i..." % k

        # random parameters
        nb_filters = np.random.randint(1, 10)
        stack_size = np.random.randint(1, 10)
        filter_height = np.random.randint(1, 10)
        wide = 1  # np.random.randint(2)
        emb_dim = np.random.randint(1, 20)

        # hidden layer
        conv_layer = convolution.Conv1DLayer(nb_filters, stack_size, filter_height, wide, emb_dim, 'conv1d_layer')
        filters = conv_layer.filters.get_value()
        bias = conv_layer.bias.get_value()

        for i in xrange(20):
            print "%i" % i,

            # tests for dimension 1, 2, 3 and 4
            input = T.tensor4('input_test')
            input_value = np.random.rand(
                1,
                stack_size,
                np.random.randint(1, 30),
                emb_dim
            ).astype(floatX)
            while (not wide) and input_value.shape[2] < filters.shape[3]:
                input_value = np.random.rand(
                    1,
                    stack_size,
                    np.random.randint(1, 30),
                    emb_dim
                ).astype(floatX)
            output = conv_layer.link(input)

            expected_value_shape = (1, nb_filters, input_value.shape[2] + (filters.shape[3] - 1) * (1 if wide else -1), emb_dim)
            expected_value = np.zeros(expected_value_shape)
            # filter : (emb_dim, nb_filters, stack_size, filter_height, 1)

            for i in xrange(nb_filters):
                for j in xrange(stack_size):
                    for k in xrange(emb_dim):
                        # print ""
                        # print wide, input_value.shape, filters.shape
                        # print expected_value.shape
                        # print input_value[0][j][:,k].shape
                        # print filters[k][i][j].flatten().shape
                        # print np.convolve(input_value[0][j][:,k], filters[k][i][j].flatten(), mode = "full" if wide else "valid").shape
                        # print expected_value[0][i][:,k].shape
                        expected_value[0][i][:, k] += convolve(
                            input_value[0, j, :, k],
                            filters[k, i, j].flatten(),
                            mode="full" if wide else "valid"
                        )

            # print expected_value[0][0]
            # print output.eval({input:input_value})[0][0]
            # return

            for i in xrange(nb_filters):
                for j in xrange(emb_dim):
                    expected_value[:, i, :, j] += bias[i, j]
            expected_value = np.tanh(expected_value)
            assert expected_value.shape == expected_value_shape

            np.testing.assert_array_almost_equal(
                output.eval({input: input_value}),
                expected_value,
                decimal=4
            )

        print "OK"

    print "All tests ran successfully for 1D Convolution Layer."


def test_conv2d_layer():

    print "Testing 2D convolutional layer..."

    for k in xrange(5):
        print "Layer %i..." % k

        # random parameters
        nb_filters = 1
        stack_size = np.random.randint(1, 10)
        filter_height = np.random.randint(1, 10)
        filter_width = np.random.randint(1, 10)
        wide = np.random.randint(2)

        # hidden layer
        conv_layer = convolution.Conv2DLayer(nb_filters, stack_size, filter_height, filter_width, wide, "conv_layer")
        filter = conv_layer.filters.get_value()[0]

        for i in xrange(20):
            print "%i" % i,

            # tests for dimension 1, 2, 3 and 4
            input = T.tensor4('input_test')
            input_value = np.random.rand(
                stack_size,
                np.random.randint(1, 20),
                np.random.randint(1, 20)
            ).astype(floatX)
            while not wide and (input_value.shape[1] < filter.shape[1] or input_value.shape[2] < filter.shape[2]):
                input_value = np.random.rand(
                    stack_size,
                    np.random.randint(1, 20),
                    np.random.randint(1, 20)
                ).astype(floatX)
            input_value_4d = np.array([input_value])

            output = conv_layer.link(input)

            expected_value = convolve2d(input_value[0], filter[0], mode="full" if wide else "valid")
            for i in xrange(1, stack_size):
                expected_value += convolve2d(input_value[i], filter[i], mode="full" if wide else "valid")
            expected_value += conv_layer.bias.get_value()
            expected_value = np.tanh(expected_value)

            assert expected_value.shape[0] == input_value.shape[1] + (filter.shape[1] - 1) * (1 if wide else -1)
            assert expected_value.shape[1] == input_value.shape[2] + (filter.shape[2] - 1) * (1 if wide else -1)
            np.testing.assert_array_almost_equal(
                output.eval({input: input_value_4d})[0][0],
                expected_value,
                decimal=6
            )

        print "OK"

    print "All tests ran successfully for 2D Convolution Layer."


def test_conv1d_layer_kmax_pooling():

    print "Testing 1D convolutional layer with k-max pooling..."

    for k in xrange(5):
        print "Layer %i..." % k

        # random parameters
        nb_filters = 1  # np.random.randint(1, 10)
        stack_size = np.random.randint(1, 10)
        filter_height = np.random.randint(1, 10)
        wide = np.random.randint(2)
        emb_dim = np.random.randint(5, 20)

        # hidden layer
        conv_layer = convolution.Conv1DLayerKMaxPooling(
            nb_filters,
            stack_size,
            filter_height,
            wide,
            emb_dim,
            'conv1d_layer'
        )
        filters = conv_layer.conv1d_layer.filters.get_value()
        bias = conv_layer.conv1d_layer.bias.get_value()

        for i in xrange(20):
            print "%i" % i,

            # tests for dimension 1, 2, 3 and 4
            input = T.tensor4('input_test')

            input_value = np.random.rand(
                1,
                stack_size,
                np.random.randint(20, 30),
                emb_dim
            ).astype(floatX)
            while (not wide) and input_value.shape[2] < filters.shape[3]:
                input_value = np.random.rand(
                    1,
                    stack_size,
                    np.random.randint(20, 30),
                    emb_dim
                ).astype(floatX)

            # k_max = T.iscalar('k_max')
            k_max_value = np.random.randint(1, 10)

            pooling.set_k_max(conv_layer, k_max_value, 1, 1, input.shape[2])
            output = conv_layer.link(input)

            expected_value_shape = (1, nb_filters, input_value.shape[2] + (filters.shape[3] - 1) * (1 if wide else -1), emb_dim)
            expected_value = np.zeros(expected_value_shape)
            # filter : (emb_dim, nb_filters, stack_size, filter_height, 1)

            for i in xrange(nb_filters):
                for j in xrange(stack_size):
                    for k in xrange(emb_dim):
                        # print ""
                        # print wide, input_value.shape, filters.shape
                        # print expected_value.shape
                        # print input_value[0][j][:,k].shape
                        # print filters[k][i][j].flatten().shape
                        # print np.convolve(input_value[0][j][:,k], filters[k][i][j].flatten(), mode = "full" if wide else "valid").shape
                        # print expected_value[0][i][:,k].shape
                        expected_value[0][i][:, k] += convolve(
                            input_value[0, j, :, k],
                            filters[k, i, j].flatten(),
                            mode="full" if wide else "valid"
                        )

            # print expected_value[0][0]
            # print output.eval({input:input_value})[0][0]
            # return

            expected_value = kmax_pooling(np.array(expected_value), k_max_value)

            for i in xrange(nb_filters):
                for j in xrange(emb_dim):
                    expected_value[:, i, :, j] += bias[i, j]
            expected_value = np.tanh(expected_value)

            assert expected_value.shape[:2] == (1, nb_filters)
            assert expected_value.shape[2] in [k_max_value, input_value.shape[2] + (filters.shape[3] - 1) * (1 if wide else -1)]
            assert expected_value.shape[3] == input_value.shape[3]

            # print wide, k_max_value, input_value.shape, filters.shape, expected_value.shape
            np.testing.assert_array_almost_equal(
                output.eval({input: input_value}),
                expected_value,
                decimal=4
            )

        print "OK"

    print "All tests ran successfully for 1D Convolution Layer with k-max pooling."


def test_conv2d_layer_kmax_pooling():

    rng = np.random.RandomState(1)
    print "Testing 2D Convolutional Layer with k-max Pooling..."

    for k in xrange(5):
        print "Layer %i..." % k

        # random parameters
        nb_filters = 1
        stack_size = rng.randint(1, 10)
        filter_height = rng.randint(1, 10)
        filter_width = rng.randint(1, 10)
        wide = rng.randint(2)

        # hidden layer
        conv_layer = convolution.Conv2DLayerKMaxPooling(
            nb_filters,
            stack_size,
            filter_height,
            filter_width,
            wide,
            "conv_layer_kmax_pooling"
        )
        filters = conv_layer.conv2d_layer.filters.get_value()[0]
        # bias = conv_layer.conv2d_layer.bias.get_value()

        for i in xrange(20):
            print "%i" % i,

            # tests for dimension 1, 2, 3 and 4
            input = T.tensor4('input_test')
            input_value = rng.rand(stack_size, rng.randint(20, 30), rng.randint(5, 20)).astype(floatX)
            while not wide and (input_value.shape[1] < filters.shape[1] or input_value.shape[2] < filters.shape[2]):
                input_value = rng.rand(stack_size, rng.randint(10, 20), rng.randint(5, 20)).astype(floatX)
            input_value_4d = np.array([input_value])
            # k_max = T.iscalar('k_max')
            k_max_value = rng.randint(1, 10)

            pooling.set_k_max(conv_layer, k_max_value, 1, 1, input.shape[2])
            output = conv_layer.link(input)

            expected_value = convolve2d(input_value[0], filters[0], mode="full" if wide else "valid")
            for i in xrange(1, stack_size):
                expected_value += convolve2d(input_value[i], filters[i], mode="full" if wide else "valid")

            expected_value = kmax_pooling(np.array([[expected_value]]), k_max_value)

            expected_value += conv_layer.conv2d_layer.bias.get_value()
            expected_value = np.tanh(expected_value)

            assert expected_value.shape[:2] == (1, 1)
            assert expected_value.shape[2] in [k_max_value, input_value.shape[1] + (filters.shape[1] - 1) * (1 if wide else -1)]
            assert expected_value.shape[3] == input_value.shape[2] + (filters.shape[2] - 1) * (1 if wide else -1)

            np.testing.assert_array_almost_equal(
                output.eval({input: input_value_4d}),
                expected_value,
                decimal=4
            )

        print "OK"

    print "All tests ran successfully for 2D Convolution Layer with k-max Pooling."


"""
"""

test_hidden_layer()
test_embedding_layer()
test_rnn()
test_lstm()
test_kmax_pooling_layer_1()
test_kmax_pooling_layer_2()
test_conv1d_layer()
test_conv2d_layer()
test_conv1d_layer_kmax_pooling()
test_conv2d_layer_kmax_pooling()


exit()





start = timeit.default_timer()
test_kmax_pooling_layer_1()
stop = timeit.default_timer()
print stop - start

start = timeit.default_timer()
test_kmax_pooling_layer_2()
stop = timeit.default_timer()
print stop - start
