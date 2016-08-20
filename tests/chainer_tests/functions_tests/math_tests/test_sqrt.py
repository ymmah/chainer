import numpy

import chainer.functions as F
from chainer import testing


#
# sqrt

class TestSqrt(testing.UnaryFunctionTestBase):

    def func(self):
        return F.sqrt

    def make_data(self, dtype, shape):
        x = numpy.random.uniform(0.1, 1, shape).astype(dtype)
        gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
        return x, gy


#
# rsqrt

class TestRsqrt(testing.UnaryFunctionTestBase):

    def func(self):
        return F.rsqrt

    def func_expected(self):
        def rsqrt(x, dtype=numpy.float32):
            return numpy.reciprocal(numpy.sqrt(x, dtype=dtype))
        return rsqrt

    def make_data(self, dtype, shape):
        x = numpy.random.uniform(0.1, 1, shape).astype(dtype)
        gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
        return x, gy


testing.run_module(__name__, __file__)
