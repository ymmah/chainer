import numpy

import chainer.functions as F
from chainer import testing


def make_data(dtype, shape):
    x = numpy.random.uniform(0.1, 1, shape).astype(dtype)
    gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
    return x, gy


#
# sqrt

TestSqrt = testing.unary_function_test(F.sqrt, make_data=make_data)


#
# rsqrt

def rsqrt(x, dtype=numpy.float32):
    return numpy.reciprocal(numpy.sqrt(x, dtype=dtype))

TestRsqrt = testing.unary_function_test(F.rsqrt, rsqrt, make_data)


testing.run_module(__name__, __file__)
