import unittest

import chainer.functions as F
from chainer import testing


#
# sin

@testing.math_function_test(F.sin)
class TestSin(unittest.TestCase):
    pass


#
# cos

@testing.math_function_test(F.cos)
class TestCos(unittest.TestCase):
    pass


#
# tan

@testing.math_function_test(F.tan)
class TestTan(unittest.TestCase):
    pass


testing.run_module(__name__, __file__)
