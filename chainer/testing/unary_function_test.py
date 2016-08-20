import unittest

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.testing import attr
from chainer.testing import condition


class UnaryFunctionTestBase(unittest.TestCase):
    """To be described."""

    def func(self):
        raise NotImplementedError()

    def func_expected(self):
        name = self.func().__name__
        return getattr(numpy, name)

    def func_class(self):
        name = self.func().__name__
        name = name[0].upper() + name[1:]
        return getattr(F, name, None)

    def make_data(self, dtype, shape):
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)
        gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
        return x, gy

    def backward_options(self, dtype):
        if dtype == numpy.float16:
            return {'eps': 2 ** -4, 'atol': 2 ** -4, 'rtol': 2 ** -4,
                    'dtype': numpy.float64}
        else:
            return {}

    def check_forward(self, gpu, dtype, shape):
        from chainer import testing  # Import here to avoid mutual import.
        x_data, _ = self.make_data(dtype, shape)
        if gpu:
            x = chainer.Variable(cuda.to_gpu(x_data))
        else:
            x = chainer.Variable(x_data)
        y = self.func()(x)
        self.assertEqual(y.data.dtype, dtype)
        y_expected = self.func_expected()(x_data, dtype=dtype)
        testing.assert_allclose(y_expected, y.data, atol=1e-4, rtol=1e-4)

    @condition.retry(3)
    def test_forward_cpu_fp16(self):
        self.check_forward(False, numpy.float16, (3, 2))
        self.check_forward(False, numpy.float16, ())

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_fp16(self):
        self.check_forward(True, numpy.float16, (3, 2))
        self.check_forward(True, numpy.float16, ())

    @condition.retry(3)
    def test_forward_cpu_fp32(self):
        self.check_forward(False, numpy.float32, (3, 2))
        self.check_forward(False, numpy.float32, ())

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_fp32(self):
        self.check_forward(True, numpy.float32, (3, 2))
        self.check_forward(True, numpy.float32, ())

    @condition.retry(3)
    def test_forward_cpu_fp64(self):
        self.check_forward(False, numpy.float64, (3, 2))
        self.check_forward(False, numpy.float64, ())

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_fp64(self):
        self.check_forward(True, numpy.float64, (3, 2))
        self.check_forward(True, numpy.float64, ())

    def check_backward(self, gpu, dtype, shape):
        from chainer import gradient_check  # Import here to avoid mutual imp.
        x_data, y_grad = self.make_data(dtype, shape)
        if gpu:
            x_data = cuda.to_gpu(x_data)
            y_grad = cuda.to_gpu(y_grad)
        gradient_check.check_backward(
            self.func(), x_data, y_grad, **self.backward_options(dtype))

    @condition.retry(3)
    def test_backward_cpu_fp16(self):
        self.check_backward(False, numpy.float16, (3, 2))
        self.check_backward(False, numpy.float16, ())

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_fp16(self):
        self.check_backward(True, numpy.float16, (3, 2))
        self.check_backward(True, numpy.float16, ())

    @condition.retry(3)
    def test_backward_cpu_fp32(self):
        self.check_backward(False, numpy.float32, (3, 2))
        self.check_backward(False, numpy.float32, ())

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_fp32(self):
        self.check_backward(True, numpy.float32, (3, 2))
        self.check_backward(True, numpy.float32, ())

    @condition.retry(3)
    def test_backward_cpu_fp64(self):
        self.check_backward(False, numpy.float64, (3, 2))
        self.check_backward(False, numpy.float64, ())

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_fp64(self):
        self.check_backward(True, numpy.float64, (3, 2))
        self.check_backward(True, numpy.float64, ())

    def test_label(self):
        klass = self.func_class()
        if klass is not None:
            self.assertEqual(klass().label, self.func().__name__)
