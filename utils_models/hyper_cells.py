import tensorflow as tf
import numpy as np
from tensorflow.python.keras import activations
from tensorflow.python.keras.utils import tf_utils
from tensorflow.layers import batch_normalization as bn
from tensorflow.layers import dropout as dr


class HyperLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units_main, num_units_hyper, dim_z, keep_prob=0.9, forget_bias=1.0, state_is_tuple=True, activation=None,
                 reuse=None, trainable=True, name=None, dtype=tf.float32, **kwargs):
        """
        A basic version of TF Hyper LSTM network implemented based on the tf RNNCell
        :param num_units: number of units in each nodes
        :param forget_bias: bias added to forget gate
        :param state_is_tuple: output type of state size
        :param activation: activation funtion
        :param reuse: reuse or not
        :param trainable: is the cell trainable
        :param name: variable scope name
        :param dtype: data type
        :param kwargs:  Dict, keyword named properties for common layer attributes, like
        `trainable` etc when constructing the cell from configs of get_config()
        """
        super().__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        self.num_main = num_units_main
        self.num_hyper = num_units_hyper
        self.dim_z = dim_z
        self.forget_b = forget_bias
        self.state_t = state_is_tuple
        self.keep_prob = keep_prob
        self.d_type = dtype

        if activation:
            self.act = activations.get(activation)
        else:
            self.act = tf.tanh
        if name:
            self.name_ = name
        else:
            self.name_ = "HyperLSTM"

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        """
        Define variables for the LSTM cell.
        w_hhs: z=wh+(b)
        w_hzs: d=wz+(b)
        w_ys: y=LN(dwh+dwx+dz)
        :return: None
        """
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % str(inputs_shape))
        inputs_dim = inputs_shape[-1]
        self.inputs_dim = inputs_dim

        self.gates = ['i', 'g', 'f', 'o']

        self.basic_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.num_hyper,
                                                  forget_bias=self.forget_b,
                                                  state_is_tuple=self.state_t,
                                                  activation=self.act,
                                                  dtype=self.d_type)
        self.basic_cell.build(inputs_shape)

        with tf.variable_scope("cal_Z"):
            self.w_hhs = tf.get_variable(name="W_y_h_hat",
                                         shape=[self.num_hyper, self.dim_z * 3 * 4],
                                         initializer=tf.orthogonal_initializer(), dtype=self.d_type)
            self.b_hhs = tf.get_variable(name="b_y_h_hat",
                                         shape=[4, self.dim_z * 2],
                                         initializer=tf.zeros_initializer(),
                                         dtype=self.d_type)

        with tf.variable_scope("cal_final_W"):
            self.w_hz = tf.get_variable(name="W_y_hz",
                                        shape=[4, self.num_main, self.dim_z, self.num_hyper],
                                        initializer=tf.orthogonal_initializer(),
                                        dtype=self.d_type)
            self.w_xz = tf.get_variable(name="W_y_xz",
                                        shape=[4, self.num_main, self.dim_z, self.inputs_dim],
                                        initializer=tf.orthogonal_initializer(),
                                        dtype=self.d_type)
            self.w_bz = tf.get_variable(name="W_y_bz",
                                        shape=[4, self.num_main, self.dim_z],
                                        initializer=tf.orthogonal_initializer(),
                                        dtype=self.d_type)
            self.b0 = tf.get_variable(name="W_y_b",
                                      shape=[4, self.num_main],
                                      initializer=tf.zeros_initializer(),
                                      dtype=self.d_type)

        self.built = True

    @property
    def state_size(self):
        """States of hyper cell and real cell"""
        if self.state_t:
            return tf.contrib.rnn.LSTMStateTuple(self.num_main + self.num_hyper, self.num_main + self.num_hyper)
        else:
            return 2 * self.num_main + 2 * self.num_hyper

    @property
    def output_size(self):
        return self.num_main

    def call(self, inputs, state):
        """z->d->y"""
        if self.state_t:
            # (N, M)
            _c, _h = state
            c_ = tf.slice(_c, [0, 0], [-1, self.num_main])
            c_hat_ = tf.slice(_c, [0, self.num_main], [-1, self.num_hyper])
            h_ = tf.slice(_h, [0, self.num_main], [-1, self.num_main])
            h_hat_ = tf.slice(_h, [0, self.num_main], [-1, self.num_hyper])
        else:
            c_ = tf.slice(state, [0, 0], [-1, self.num_main])
            h_ = tf.slice(state, [0, self.num_main], [-1, self.num_main])
            c_hat_ = tf.slice(state, [0, self.num_main * 2], [-1, self.num_hyper])
            h_hat_ = tf.slice(state, [0, self.num_main * 2 + self.num_hyper], [-1, self.num_hyper])

        _, state_hat = self.basic_cell.call(inputs, tf.contrib.rnn.LSTMStateTuple(c_hat_, h_hat_))

        zs = tf.matmul(h_hat_, self.w_hhs)

        # Calculate final weights
        h_hat_, inputs = tf.expand_dims(h_hat_, 2), tf.expand_dims(inputs, 2)
        zi, zg, zf, zo = tf.split(zs, [self.dim_z * 3] * 4, axis=1)
        zss = [zi, zg, zf, zo]
        ys = []
        for i, _ in enumerate(self.gates):
            zyh, zyx, zyb = tf.split(zss[i], [self.dim_z] * 3, axis=1)
            bh, bx = tf.split(self.b_hhs[i], [self.dim_z] * 2)

            zyh, zyx = tf.expand_dims(zyh + bh, 0), tf.expand_dims(zyx + bx, 0)
            zyh, zyx = tf.tile(zyh, [self.num_main, 1, 1]), tf.tile(zyx, [self.num_main, 1, 1])

            wyh = tf.transpose(tf.matmul(zyh, self.w_hz[i]), [1, 0, 2])
            wyx = tf.transpose(tf.matmul(zyx, self.w_xz[i]), [1, 0, 2])
            by = tf.matmul(zyb, self.w_bz[i], transpose_b=True) + self.b0[i]

            y = tf.squeeze(tf.matmul(wyh, h_hat_) + tf.matmul(wyx, inputs)) + by
            ys.append(bn(y))

        i, g, f, o = ys

        c = tf.sigmoid(f) * c_ + tf.sigmoid(i) * dr(tf.tanh(g), rate=1 - self.keep_prob)
        h = tf.sigmoid(o) * tf.tanh(bn(c))

        if self.state_t:
            (c_hat, h_hat) = state_hat
            _c, _h = tf.concat([c, c_hat], axis=1), tf.concat([h, h_hat], axis=1)
            new_state = tf.contrib.rnn.LSTMStateTuple(_c, _h)
        else:
            c_hat, h_hat = tf.split(state_hat, 2)
            _c, _h = tf.concat([c, c_hat], axis=1), tf.concat([h, h_hat], axis=1)
            new_state = tf.contrib.rnn.LSTMStateTuple(_c, _h)

        return h, new_state


class HyperLSTMCell_Efficient(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units_main, num_units_hyper, dim_z, keep_prob=0.9, forget_bias=1.0, state_is_tuple=True, activation=None,
                 reuse=None, trainable=True, name=None, dtype=tf.float32, **kwargs):
        """
        A memory efficient version of TF Hyper LSTM network implemented based on the tf RNNCell
        :param num_units: number of units in each nodes
        :param forget_bias: bias added to forget gate
        :param state_is_tuple: output type of state size
        :param activation: activation funtion
        :param reuse: reuse or not
        :param trainable: is the cell trainable
        :param name: variable scope name
        :param dtype: data type
        :param kwargs:  Dict, keyword named properties for common layer attributes, like
        `trainable` etc when constructing the cell from configs of get_config()
        """
        super().__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        self.num_main = num_units_main
        self.num_hyper = num_units_hyper
        self.dim_z = dim_z
        self.forget_b = forget_bias
        self.state_t = state_is_tuple
        self.keep_prob = keep_prob
        self.d_type = dtype

        if activation:
            self.act = activations.get(activation)
        else:
            self.act = tf.tanh
        if name:
            self.name_ = name
        else:
            self.name_ = "HyperLSTM"

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        """
        Define variables for the LSTM cell.
        w_hhs: z=wh+(b)
        w_hzs: d=wz+(b)
        w_ys: y=LN(dwh+dwx+dz)
        :return: None
        """
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % str(inputs_shape))
        inputs_dim = inputs_shape[-1]
        self.inputs_dim = inputs_dim

        self.gates = ['i', 'g', 'f', 'o']

        self.basic_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.num_hyper,
                                                  forget_bias=self.forget_b,
                                                  state_is_tuple=self.state_t,
                                                  activation=self.act,
                                                  dtype=self.d_type)
        self.basic_cell.build(inputs_shape)

        with tf.variable_scope("cal_Z"):
            self.w_hhs = tf.get_variable(name="W_y_h_hat",
                                         shape=[self.num_hyper, self.dim_z * 3 * 4],
                                         initializer=tf.orthogonal_initializer(), dtype=self.d_type)
            self.b_hhs = tf.get_variable(name="b_y_h_hat",
                                         shape=[4, self.dim_z * 2],
                                         initializer=tf.zeros_initializer(),
                                         dtype=self.d_type)

        with tf.variable_scope("cal_final_W"):
            self.w_hz = tf.get_variable(name="W_y_hz",
                                        shape=[4, self.dim_z, self.num_main],
                                        initializer=tf.orthogonal_initializer(),
                                        dtype=self.d_type)
            self.w_xz = tf.get_variable(name="W_y_xz",
                                        shape=[4, self.dim_z, self.num_main],
                                        initializer=tf.orthogonal_initializer(),
                                        dtype=self.d_type)
            self.w_bz = tf.get_variable(name="W_y_bz",
                                        shape=[4, self.dim_z, self.num_main],
                                        initializer=tf.orthogonal_initializer(),
                                        dtype=self.d_type)
            self.w_yh = tf.get_variable(name="W_y_h",
                                        shape=[4, self.num_hyper, self.num_main],
                                        initializer=tf.orthogonal_initializer(),
                                        dtype=self.d_type)
            self.w_yx = tf.get_variable(name="W_y_x",
                                        shape=[4, self.inputs_dim, self.num_main],
                                        initializer=tf.orthogonal_initializer(),
                                        dtype=self.d_type)

        self.built = True

    @property
    def state_size(self):
        """States of hyper cell and real cell"""
        if self.state_t:
            return tf.contrib.rnn.LSTMStateTuple(self.num_main + self.num_hyper, self.num_main + self.num_hyper)
        else:
            return 2 * self.num_main + 2 * self.num_hyper

    @property
    def output_size(self):
        return self.num_main

    def call(self, inputs, state):
        """z->d->y"""
        if self.state_t:
            _c, _h = state
            c_ = tf.slice(_c, [0, 0], [-1, self.num_main])
            c_hat_ = tf.slice(_c, [0, self.num_main], [-1, self.num_hyper])
            h_ = tf.slice(_h, [0, self.num_main], [-1, self.num_main])
            h_hat_ = tf.slice(_h, [0, self.num_main], [-1, self.num_hyper])
        else:
            c_ = tf.slice(state, [0, 0], [-1, self.num_main])
            h_ = tf.slice(state, [0, self.num_main], [-1, self.num_main])
            c_hat_ = tf.slice(state, [0, self.num_main * 2], [-1, self.num_hyper])
            h_hat_ = tf.slice(state, [0, self.num_main * 2 + self.num_hyper], [-1, self.num_hyper])

        _, state_hat = self.basic_cell.call(inputs, tf.contrib.rnn.LSTMStateTuple(c_hat_, h_hat_))

        zs = tf.matmul(h_hat_, self.w_hhs)

        # Calculate final weights
        zi, zg, zf, zo = tf.split(zs, [self.dim_z * 3] * 4, axis=1)
        zss = [zi, zg, zf, zo]
        ys = []
        for i, _ in enumerate(self.gates):
            zyh, zyx, zyb = tf.split(zss[i], [self.dim_z] * 3, axis=1)
            bh, bx = tf.split(self.b_hhs[i], [self.dim_z] * 2)
            zyh, zyx = zyh + bh, zyx + bx

            dyh, dyx, by = tf.matmul(zyh, self.w_hz[i]), tf.matmul(zyh, self.w_xz[i]), tf.matmul(zyb, self.w_bz[i])

            y = dyh * tf.matmul(h_hat_, self.w_yh[i]) + dyx * tf.matmul(inputs, self.w_yx[i]) + by
            ys.append(bn(y))

        i, g, f, o = ys

        c = tf.sigmoid(f) * c_ + tf.sigmoid(i) * dr(tf.tanh(g), rate=1 - self.keep_prob)
        h = tf.sigmoid(o) * tf.tanh(bn(c))

        if self.state_t:
            (c_hat, h_hat) = state_hat
            _c, _h = tf.concat([c, c_hat], axis=1), tf.concat([h, h_hat], axis=1)
            new_state = tf.contrib.rnn.LSTMStateTuple(_c, _h)
        else:
            c_hat, h_hat = tf.split(state_hat, 2)
            _c, _h = tf.concat([c, c_hat], axis=1), tf.concat([h, h_hat], axis=1)
            new_state = tf.contrib.rnn.LSTMStateTuple(_c, _h)

        return h, new_state
