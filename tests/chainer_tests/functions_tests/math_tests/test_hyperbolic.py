import unittest

import chainer.functions as F
from chainer import testing


#
# cosh

@testing.math_function_test(F.cosh)
class TestCosh(unittest.TestCase):
    pass


#
# sinh

@testing.math_function_test(F.sinh)
class TestSinh(unittest.TestCase):
    pass


testing.run_module(__name__, __file__)
