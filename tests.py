import numpy as np
import theano
import theano.tensor as T
import convolution
import utils
import layer
import network
import timeit
reload(convolution)
reload(layer)
reload(network)
reload(utils)

input = T.tensor4('input_test')
k_max = T.iscalar('')
kmax_pooling_layer_old = convolution.kmax_Pooling_Layer_1(k_max)
output1 = kmax_pooling_layer_old.link(input)
kmax_pooling_layer = convolution.kmax_Pooling_Layer_2(k_max)
output2 = kmax_pooling_layer.link(input)

f1 = theano.function(
    inputs=[k_max, input],
    outputs=output1
)
f2 = theano.function(
    inputs=[k_max, input],
    outputs=output2
)

rng = np.random.RandomState(1)

start = timeit.default_timer()

for _ in xrange(500):
    input_value = rng.rand(np.random.randint(1, 20), rng.randint(1, 20), rng.randint(10, 20), rng.randint(1, 100)).astype(theano.config.floatX)
    k_max_value = rng.randint(1, 10)
    #expected_value1 = output1.eval({k_max:k_max_value, input:input_value})
    #expected_value1 = f1(k_max_value, input_value)
    #expected_value2 = output2.eval({k_max:k_max_value, input:input_value})
    expected_value2 = f2(k_max_value, input_value)
    #np.testing.assert_almost_equal(expected_value1, expected_value2)

stop = timeit.default_timer()
print stop - start
