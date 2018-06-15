#-*- encoding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np


from ..base import optimizers
from ..base import losses


class Sequential(object):
    """Linear stack of layers.
    # Arguments
        layers: list of layers to add to the model.
    # Note
        The first layer passed to a Sequential model
        should have a defined input shape. What that
        means is that it should have received an `input_shape`
        or `batch_input_shape` argument,
        or for some type of layers (recurrent, Dense...)
        an `input_dim` argument.
    # Example
        ```python
            model = Sequential()
            # first layer must have a defined input shape
            model.add(Dense(32, input_dim=500))
            # afterwards, Keras does automatic shape inference
            model.add(Dense(32))
            # also possible (equivalent to the above):
            model = Sequential()
            model.add(Dense(32, input_shape=(500,)))
            model.add(Dense(32))
            # also possible (equivalent to the above):
            model = Sequential()
            # here the batch dimension is None,
            # which means any batch size will be accepted by the model.
            model.add(Dense(32, batch_input_shape=(None, 500)))
            model.add(Dense(32))
        ```
    """
    def __init__(self):
        self.layers = list()

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer='sgd', **kwargs):
        """Configures the model for training.
        # Arguments
            loss: str (name of objective function) or objective function.
                See [losses](/losses).
                If the model has multiple outputs, you can use a different loss
                on each output by passing a dictionary or a list of losses.
                The loss value that will be minimized by the model
                will then be the sum of all individual losses.
            optimizer: str (name of optimizer) or optimizer object.
                See [optimizers](/optimizers).
        # Raises
            ValueError: In case of invalid arguments for
                `optimizer`, `loss`.
        """
        loss = loss or None
        optimizer = optimizer or None
        self.optimizer = optimizers.get(optimizer)
        self.loss = losses.get(loss)

        prev_layer = None
        for layer in self.layers:
            layer.connection(prev_layer)
            prev_layer = layer


    def fit(self, X, y, epochs=100, batch_size=64, shuffle=True,
            validation_split=0., validation_data=None, verbose=1, file=sys.stdout):
        # prepare data
        X = np.asarray(X)
        y = np.asarray(y)
        train_X = X.astype(np.float64) if not np.issubdtype(np.float64, X.dtype) else X
        train_y = y.astype(np.float64) if not np.issubdtype(np.float64, y.dtype) else y

        if 1. > validation_split > 0.:
            split = int(train_y.shape[0] * validation_split)
            valid_X, valid_y = train_X[-split:], train_y[-split:]
            train_X, train_y = train_X[:-split], train_y[:-split]
        elif validation_data is not None:
            valid_X, valid_y = validation_data
            valid_X = np.asarray(valid_X)
            valid_y = np.asarray(valid_y)
        else:
            valid_X, valid_y = None, None

        iter_idx = 0
        while iter_idx < epochs:
            iter_idx += 1

            # shuffle
            if shuffle:
                seed = np.random.randint(1107)
                np.random.seed(seed)
                np.random.shuffle(train_X)
                np.random.seed(seed)
                np.random.shuffle(train_y)

            # train
            train_losses, train_predicts, train_targets = [], [], []
            total_round = train_y.shape[0] // batch_size
            if total_round != train_y.shape[0] * 1.0 / batch_size:
                total_round += 1
            for b in range(total_round):
                batch_begin = b * batch_size
                batch_end = min(batch_begin + batch_size, train_y.shape[0])
                x_batch = train_X[batch_begin:batch_end]
                y_batch = train_y[batch_begin:batch_end]

                # forward propagation
                y_pred = self.predict(x_batch, is_train=True)

                # backward propagation
                next_grad = self.loss.backward(y_pred, y_batch)
                for layer in self.layers[::-1]:
                    next_grad = layer.backward(next_grad)

                # get parameter and gradients
                params = list()
                grads = list()
                for layer in self.layers:
                    params += layer.params
                    grads += layer.grads

                # update parameters
                self.optimizer.minimize(params, grads)

                # got loss and predict
                train_losses.append(self.loss.forward(y_pred, y_batch))
                train_predicts.extend(y_pred)
                train_targets.extend(y_batch)
                if verbose == 2:
                    runout = "epoch %d/%d, batch %d/%d, train-[loss %.4f, acc %.4f]; " % (
                        iter_idx, epochs, b + 1, total_round, float(np.mean(train_losses)),
                        float(self.accuracy(train_predicts, train_targets)))
                    print(runout, file=file)

            # output train status
            runout = "epoch %d/%d, train-[loss %.4f, acc %.4f]; " % (
                iter_idx, epochs, float(np.mean(train_losses)),
                float(self.accuracy(train_predicts, train_targets)))

            if valid_X is not None and valid_y is not None:
                # valid
                valid_losses, valid_predicts, valid_targets = [], [], []
                for b in range(valid_X.shape[0] // batch_size):
                    batch_begin = b * batch_size
                    batch_end = batch_begin + batch_size
                    x_batch = valid_X[batch_begin:batch_end]
                    y_batch = valid_y[batch_begin:batch_end]

                    # forward propagation
                    y_pred = self.predict(x_batch, is_train=False)

                    # got loss and predict
                    valid_losses.append(self.loss.forward(y_pred, y_batch))
                    valid_predicts.extend(y_pred)
                    valid_targets.extend(y_batch)

                # output valid status
                runout += "valid-[loss %.4f, acc %.4f]; " % (
                    float(np.mean(valid_losses)), float(self.accuracy(valid_predicts, valid_targets)))

            if verbose > 0:
                print(runout, file=file)

    def predict(self, X, is_train=False):
        """ Calculate an output Y for the given input X. """
        x_next = X
        for layer in self.layers[:]:
            x_next = layer.forward(x_next, is_train=is_train)
        y_pred = x_next
        return y_pred

    def accuracy(self, outputs, targets):
        y_predicts = np.argmax(outputs, axis=1)
        y_targets = np.argmax(targets, axis=1)
        acc = y_predicts == y_targets
        return np.mean(acc)


class Model(object):
    """Layers with multiple input and multiple output.
    # Arguments
        input: the input layer
        output: the output layer
    # Note
        The first layer passed to a Model model
        should have a defined input shape. What that
        means is that it should have received an `input_shape`
        or `batch_input_shape` argument,
        or for some type of layers (recurrent, Dense...)
        an `input_dim` argument.
    # TODO
        Add the multiple input and multiple output form, now it's similar
        to Sequential model.
    # Example
        ```python
            # first layer must have a defined input shape
            input = Input(input_shape=(input_dim, ))
            dense_1 = Dense(300, activator='selu')(input)
            dropout_1 = Dropout(0.2)(dense_1)
            softmax_1 = Softmax(label_size)(dropout_1)
            model = Model(input, softmax_1)
            # also possible (equivalent to the above):
            minput = Dense(32, input_dim=60)
            output = Dense(32)(input)
            model = Model(input, output)
        ```
    """
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def compile(self, loss, optimizer='sgd', **kwargs):
        """Configures the model for training.
        # Arguments
            loss: str (name of objective function) or objective function.
                See [losses](/losses).
                If the model has multiple outputs, you can use a different loss
                on each output by passing a dictionary or a list of losses.
                The loss value that will be minimized by the model
                will then be the sum of all individual losses.
            optimizer: str (name of optimizer) or optimizer object.
                See [optimizers](/optimizers).
        # Raises
            ValueError: In case of invalid arguments for
                `optimizer`, `loss`.
        """
        loss = loss or None
        optimizer = optimizer or None
        self.optimizer = optimizers.get(optimizer)
        self.loss = losses.get(loss)

    def fit(self, X, y, epochs=100, batch_size=64, shuffle=True,
            validation_split=0., validation_data=None, verbose=1, file=sys.stdout):
        # prepare data
        X = np.asarray(X)
        y = np.asarray(y)
        train_X = X.astype(np.float64) if not np.issubdtype(np.float64, X.dtype) else X
        train_y = y.astype(np.float64) if not np.issubdtype(np.float64, y.dtype) else y

        if 1. > validation_split > 0.:
            split = int(train_y.shape[0] * validation_split)
            valid_X, valid_y = train_X[-split:], train_y[-split:]
            train_X, train_y = train_X[:-split], train_y[:-split]
        elif validation_data is not None:
            valid_X, valid_y = validation_data
            valid_X = np.asarray(valid_X)
            valid_y = np.asarray(valid_y)
        else:
            valid_X, valid_y = None, None

        iter_idx = 0
        while iter_idx < epochs:
            iter_idx += 1

            # shuffle
            if shuffle:
                seed = np.random.randint(1107)
                np.random.seed(seed)
                np.random.shuffle(train_X)
                np.random.seed(seed)
                np.random.shuffle(train_y)

            # train
            train_losses, train_predicts, train_targets = [], [], []
            total_round = train_y.shape[0] // batch_size
            if total_round != train_y.shape[0] * 1.0 / batch_size:
                total_round += 1
            for b in range(total_round):
                batch_begin = b * batch_size
                batch_end = min(batch_begin + batch_size, train_y.shape[0])
                x_batch = train_X[batch_begin:batch_end]
                y_batch = train_y[batch_begin:batch_end]

                # forward propagation
                y_pred = self.predict(x_batch, is_train=True)

                # backward propagation
                next_grad = self.loss.backward(y_pred, y_batch)
                pre_layer = self.output
                while pre_layer is not None:
                    next_grad = pre_layer.backward(next_grad)
                    pre_layer = pre_layer.pre_layer

                # get parameter and gradients
                params = list()
                grads = list()
                pre_layer = self.output
                while pre_layer is not None:
                    params += pre_layer.params
                    grads += pre_layer.grads
                    pre_layer = pre_layer.pre_layer

                # update parameters
                self.optimizer.minimize(params, grads)

                # got loss and predict
                train_losses.append(self.loss.forward(y_pred, y_batch))
                train_predicts.extend(y_pred)
                train_targets.extend(y_batch)
                if verbose == 2:
                    runout = "epoch %d/%d, batch %d/%d, train-[loss %.4f, acc %.4f]; " % (
                        iter_idx, epochs, b + 1, total_round, float(np.mean(train_losses)),
                        float(self.accuracy(train_predicts, train_targets)))
                    print(runout, file=file)

            # output train status
            runout = "epoch %d/%d, train-[loss %.4f, acc %.4f]; " % (
                iter_idx, epochs, float(np.mean(train_losses)),
                float(self.accuracy(train_predicts, train_targets)))

            if valid_X is not None and valid_y is not None:
                # valid
                valid_losses, valid_predicts, valid_targets = [], [], []
                for b in range(valid_X.shape[0] // batch_size):
                    batch_begin = b * batch_size
                    batch_end = batch_begin + batch_size
                    x_batch = valid_X[batch_begin:batch_end]
                    y_batch = valid_y[batch_begin:batch_end]

                    # forward propagation
                    y_pred = self.predict(x_batch, is_train=False)

                    # got loss and predict
                    valid_losses.append(self.loss.forward(y_pred, y_batch))
                    valid_predicts.extend(y_pred)
                    valid_targets.extend(y_batch)

                # output valid status
                runout += "valid-[loss %.4f, acc %.4f]; " % (
                    float(np.mean(valid_losses)), float(self.accuracy(valid_predicts, valid_targets)))

            if verbose > 0:
                print(runout, file=file)

    def predict(self, X, is_train=False):
        """ Calculate an output Y for the given input X. """
        layer = self.input
        X_next = X
        while layer is not None:
            X_next = layer.forward(X_next)
            layer = layer.next_layer
        return X_next

    def accuracy(self, outputs, targets):
        y_predicts = np.argmax(outputs, axis=1)
        y_targets = np.argmax(targets, axis=1)
        acc = y_predicts == y_targets
        return np.mean(acc)
