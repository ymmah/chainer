#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import print_function
import argparse
import math

import numpy as np

import chainer
from chainer import cuda
from chainer.datasets import ptb_words
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer.utils import summary


class ParallelSequenceLoader(chainer.Dataset):

    """Loader of a set of sequences starting from different positions."""
    def __init__(self, base, batchsize):
        self._base = base
        self._batchsize = batchsize

    def __len__(self):
        return len(self._base)

    def __getitem__(self, i):
        count = i // self._batchsize
        skip = i % self._batchsize * len(self) // self._batchsize
        return self._base[skip + count]


class RNNLM(chainer.Chain):

    """Recurrent neural net languabe model for penn tree bank corpus.

    This is an example of deep LSTM network for infinite length input.

    """
    def __init__(self, n_vocab, n_units, train=True):
        super(RNNLM, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units, n_vocab),
        )
        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=self.train))
        h2 = self.l2(F.dropout(h1, train=self.train))
        y = self.l3(F.dropout(h2, train=self.train))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    args = parser.parse_args()
    xp = cuda.cupy if args.gpu >= 0 else np
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()

    train = ParallelSequenceLoader(ptb_words.PTBWordsTraining(), 20)
    test = ptb_words.PTBWordsValidation()
    train_iter = train.get_batch_iterator(batchsize=20, auto_shuffle=False)

    model = L.Classifier(RNNLM(test.n_vocab, 650))
    model.compute_accuracy = False
    for param in model.params():
        param.data[:] = np.random.uniform(-0.1, 0.1, param.data.shape)
    if args.gpu >= 0:
        model.to_gpu()

    optimizer = optimizers.SGD(1)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))

    state = TrainingState(model, optimizer, train_iter)
    if args.resume:
        serializers.load_npz(args.resume, state)

    epoch = 0
    loss = 0
    t = 0
    summ = summary.Summary()
    for cur, nxt in train_iter:
        loss += model(chainer.Variable(xp.asarray(cur)),
                      chainer.Variable(xp.asarray(nxt)))
        summ.add(model.loss.data)
        t += 1
        if t % 35 == 0:
            model.zerograds()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
            loss = 0

        if t % 10000 == 0:
            print('iter {} training perplexity={:.2f}'.format(
                optimizer.t, math.exp(float(summ.mean))))
            summ = summary.Summary()
            serializers.save_npz('snapshot', state)

        if epoch != train_iter.epoch:
            epoch = train_iter.epoch
            print('evaluating...')

            test_summ = summary.Summary()
            m0 = model.copy()
            m0.predictor.reset_state()
            for cur, nxt in test.get_batch_iterator(batchsize=1, repeat=False,
                                                    auto_shuffle=False):
                loss0 = m0(chainer.Variable(xp.asarray(cur), volatile='on'),
                           chainer.Variable(xp.asarray(nxt), volatile='on'))
                test_summ.add(loss0.data)
            print('epoch {} validation perplexity={:.2f}'.format(
                epoch, math.exp(float(test_summ.mean))))


if __name__ == '__main__':
    main()
