#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net.

"""
from __future__ import print_function
import argparse

import numpy as np

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer.datasets import mnist
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer.utils import log
from chainer.utils import summary


class MnistMLP(chainer.Chain):

    """An example of multi-layer perceptron for MNIST dataset.

    This is a very simple implementation of an MLP. You can modify this code to
    build your own neural net.

    """
    def __init__(self, n_in, n_units, n_out):
        super(MnistMLP, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_out),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class MnistMLPParallel(chainer.Chain):

    """An example of model-parallel MLP.

    This chain combines four small MLPs on two different devices.

    """
    def __init__(self, n_in, n_units, n_out):
        super(MnistMLPParallel, self).__init__(
            first0=MnistMLP(n_in, n_units // 2, n_units).to_gpu(0),
            first1=MnistMLP(n_in, n_units // 2, n_units).to_gpu(1),
            second0=MnistMLP(n_units, n_units // 2, n_out).to_gpu(0),
            second1=MnistMLP(n_units, n_units // 2, n_out).to_gpu(1),
        )

    def __call__(self, x):
        # assume x is on GPU 0
        x1 = F.copy(x, 1)

        z0 = self.first0(x)
        z1 = self.first1(x1)

        # sync
        h0 = z0 + F.copy(z1, 0)
        h1 = z1 + F.copy(z0, 1)

        y0 = self.second0(F.relu(h0))
        y1 = self.second1(F.relu(h1))

        # sync
        y = y0 + F.copy(y1, 0)
        return y


class TrainingState(object):

    def __init__(self, model, optimizer, dataiter):
        self._model = model
        self._optimizer = optimizer
        self._dataiter = dataiter

    def serialize(self, serializer):
        self._model.serialize(serializer['model'])
        self._optimizer.serialize(serializer['optimizer'])
        self._dataiter.serialize(serializer['dataiter'])


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--net', '-n', choices=('simple', 'parallel'),
                        default='simple', help='Network type')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    args = parser.parse_args()

    train = mnist.MnistTraining()
    test = mnist.MnistTest()

    train_iter = train.get_batch_iterator(batchsize=100)

    if args.net == 'simple':
        model = L.Classifier(MnistMLP(784, 1000, 10))
        xp = np if args.gpu < 0 else cuda.cupy
        if xp is cuda.cupy:
            model.to_gpu(args.gpu)
    elif args.net == 'parallel':
        model = L.Classifier(MnistMLPParallel(784, 1000, 10))
        xp = cuda.cupy

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    state = TrainingState(model, optimizer, train_iter)
    if args.resume:
        serializers.load_npz(args.resume, state)

    epoch = train_iter.epoch
    summ = summary.DictSummary()
    for x, t in train_iter:
        optimizer.update(model, chainer.Variable(xp.asarray(x)),
                         chainer.Variable(xp.asarray(t)))
        summ.add({'loss': model.loss.data, 'accuracy': model.accuracy.data})

        if optimizer.t == 1:
            with open('graph.dot', 'w') as o:
                g = computational_graph.build_computational_graph(
                    (model.loss,))
                o.write(g.dump())

        if epoch != train_iter.epoch:
            epoch = train_iter.epoch
            print('epoch {} result'.format(epoch))
            print(' train:', log.str_result(summ.mean))
            summ.clear()

            test_summ = summary.DictSummary()
            for x, t in test.get_batch_iterator(batchsize=100, repeat=False,
                                                auto_shuffle=False):
                model(chainer.Variable(xp.asarray(x), volatile='on'),
                      chainer.Variable(xp.asarray(t), volatile='on'))
                test_summ.add({'loss': model.loss.data,
                               'accuracy': model.accuracy.data})
            print('  test:', log.str_result(test_summ.mean))

            serializers.save_npz('snapshot', state)

        if epoch == 20:
            break


if __name__ == '__main__':
    main()
