import unittest

import inspect
import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.testing import attr
from chainer.testing import condition


def func_class(func):
    name = func.__name__
    name = name[0].upper() + name[1:]
    return getattr(F, name, None)


def class_name(func):
    name = func.__name__
    name = name[0].upper() + name[1:]
    return 'Test' + name


def make_data_default(self, dtype, shape):
    x = numpy.random.uniform(-1, 1, shape).astype(dtype)
    gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
    return x, gy


def unary_function_test(func, func_expected=None, make_data=None):
    """Make test class for Chainer functions lifting unary numpy/cupy
    functions.

    This function makes test class for Chainer functions lifted from
    corresponding unary numpy and cupy functions, and optionally ones composed
    with such other Chainer functions. Forward and backward computations on CPU
    and GPU across parameterized ``dtype`` and ``shape`` are tested.

    Args:
        func: Required. Chainer function to be tested.
        func_expected: Optional. Function that is used on testing forward
            computation to get expected values. If not give, a corresponding
            numpy function for ``func`` is implicitly picked up from its name.
        make_data: Optional. Function that takes ``dtype`` and ``shape`` to
            return a tuple of input and gradient data. If not given, default
            input and gradient are used.

    ``func`` takes a Chainer function to be tested and usually it is enough.
    ``func_expected`` is used on testing Chainer functions composed with others
    and to give their expected values. ``make_data`` is used to customize input
    and gradient data for testing. By default, uniform distribution ranged
    [-1, 1] is used for both.

    Generated test class tests forward and backward computation for CPU and GPU
    across the following :func:`~chainer.testing.parameterize` ed parameters:

    - dtype: ``numpy.float16``, ``numpy.float32`` and ``numpy.float64``
    - shape: rank of zero and more

    Additionally, it tests the label of Chainer function class if a Chainer
    function has its corresponding function class. It looks up a Chainer
    function class in ``chainer.functions`` module from the name of the Chainer
    function.

    .. admonition:: Example

       The following code defines a test class that tests trigonometric ``sin``
       Chainer function that takes a variable with ``dtype`` of float and
       returns another with the same ``dtype``.

       >>> from chainer import functions as F
       >>> from chainer import testing
       >>>
       >>> TestSin = testing.unary_function_test(F.sin)

       We may define tests for unary Chainer functions composed with other
       Chainer functions, like ``rsqrt`` which computes reciprocal of square
       root.

       >>> from chainer import functions as F
       >>> from chainer import testing
       >>>
       >>> def rsqrt(x, dtype=numpy.float32):
       >>>     return numpy.reciprocal(numpy.sqrt(x, dtype=dtype))
       >>>
       >>> TestRsqrt = testing.unary_function_test(F.rsqrt, rsqrt)

       Here we define ``rsqrt`` function composing numpy functions to get
       expected values, passing it to the second argument of
       ``unary_function_test`` function.

       We may also customize test data to be used. The following is an example
       of testing ``sqrt`` Chainer function which we want to test in positive
       value domain leaving some margin around zero of input ``x``.

       >>> def make_data(dtype, shape):
       >>>     x = numpy.random.uniform(0.1, 1, shape).astype(dtype)
       >>>     gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
       >>>     return x, gy
       >>>
       >>> TestSqrt = testing.unary_function_test(F.sqrt, make_data=make_data)

       We define ``make_data`` function to return input and gradient ndarrays
       generated in proper value domains with given ``dtype`` and ``shape``
       parameters, then passing it to the function's ``make_data`` argument.

    """

    # Import here to avoid mutual import.
    from chainer import gradient_check
    from chainer import testing

    if func_expected is None:
        name = func.__name__
        try:
            func_expected = getattr(numpy, name)
        except AttributeError:
            raise ValueError("numpy has no function corresponding "
                             "to Chainer function '{}'.".format(func.__name__))

    if make_data is None:
        make_data = make_data_default

    def setUp(self):
        self.x, self.gy = make_data(self.dtype, self.shape)
        if self.dtype == numpy.float16:
            self.backward_options = {
                'eps': 2 ** -4, 'atol': 2 ** -4, 'rtol': 2 ** -4,
                'dtype': numpy.float64}
        else:
            self.backward_options = {}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = func(x)
        self.assertEqual(y.data.dtype, x_data.dtype)
        y_expected = func_expected(cuda.to_cpu(x_data), dtype=x_data.dtype)
        testing.assert_allclose(y_expected, y.data, atol=1e-4, rtol=1e-4)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        gradient_check.check_backward(
            func, x_data, y_grad, **self.backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def test_label(self):
        klass = func_class(func)
        if klass is not None:
            self.assertEqual(klass().label, func.__name__)

    name = class_name(func)
    klass = type(name, (unittest.TestCase,),
                 {'setUp': setUp,
                  'check_forward': check_forward,
                  'test_forward_cpu': test_forward_cpu,
                  'test_forward_gpu': test_forward_gpu,
                  'check_backward': check_backward,
                  'test_backward_cpu': test_backward_cpu,
                  'test_backward_gpu': test_backward_gpu,
                  'test_label': test_label})

    # Set the module of the class to caller test module so that parameterized
    # test classes are to be pushed into the test module.
    caller_module = inspect.getmodule(inspect.stack()[1][0])
    klass.__module__ = caller_module.__name__

    # Return parameterized class.
    return testing.parameterize(*testing.product({
            'shape': [(3, 2), ()],
            'dtype': [numpy.float16, numpy.float32, numpy.float64]
        }))(klass)
