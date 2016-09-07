import unittest

import numpy

import chainer.functions as F
from chainer import testing


def make_data(dtype, shape):
    x = numpy.random.uniform(.5, 1, shape).astype(dtype)
    gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
    return x, gy


#
# exp

@testing.math_function_test(F.exp)
class TestExp(unittest.TestCase):
    pass


#
# log

@testing.math_function_test(F.log, make_data=make_data)
class TestLog(unittest.TestCase):
    pass


#
# log2

@testing.math_function_test(F.log2, make_data=make_data)
class TestLog2(unittest.TestCase):
    pass


#
# log10

@testing.math_function_test(F.log10, make_data=make_data)
class TestLog10(unittest.TestCase):
    pass


testing.run_module(__name__, __file__)
