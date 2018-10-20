import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

class sketch_rnn:
    def __init__(self, input_shape, label_num, encoder_size, encoder_dim, encoder_restore=False, pre_train=True,
                 encoder_path=None, decoder=False, vector_shape=None, decoder_size=None, decoder_dim=None):
        self.en_size = encoder_size
        self.en_dim = encoder_dim
        self.batch_size = input_shape[0]
        self.label_num = label_num

        with tf.name_scope('placeholder'):
            self.xs = tf.placeholder(input_shape, tf.float32)
            self.labels = tf.placeholder(self.batch_size, tf.int8)
            if decoder and not vector_shape:
                self.vecs = tf.placeholder(vector_shape)

        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            # define layers
            self.en_layers_fw = [tf.nn.rnn_cell.BasicLSTMCell(encoder_dim) for _ in range(encoder_size)]
            self.en_layers_bw = [tf.nn.rnn_cell.BasicLSTMCell(encoder_dim) for _ in range(encoder_size)]

        # stack layers and define initialize states
        self.multi_en_fw = tf.nn.rnn_cell.MultiRNNCell(self.en_layers_fw)
        self.multi_en_bw = tf.nn.rnn_cell.MultiRNNCell(self.en_layers_bw)

        # run by dynamic rnn
        self.en_output, self.en_states = tf.nn.bidirectional_dynamic_rnn(self.multi_en_fw, self.multi_en_bw, self.xs)
        self.en_state = tf.stack(self.en_states, axis=1)

        if encoder_restore:
            self.restorer = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                      scope='encoder'))
            self.restorer.restore(self.sess, encoder_path)

        if pre_train:
            # for pretraining process that is to predict the next step
            self.output, self.loss, self.step = self.pre_train_process()
        elif decoder:
            # for training decoder
            self.output, self.loss, self.step = self.decoder_training(decoder_size, decoder_dim)
        else:
            # for classification layer
            self.output, self.loss, self.step = self.classification()


        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def classification(self,):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            logits = tf.layers.dense(tf.reshape(self.en_state, (self.batch_size, -1)), self.label_num, name='classify')
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
            step = tf.train.AdamOptimizer().minimize(loss)
            return logits, loss, step

    def pre_train_process(self):
        # need further edit
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            tmp = tf.layers.dense(tf.reshape(self.en_state, (self.batch_size, self.en_size, -1)),
                                  units=1000, activation=tf.nn.relu, name='FC_en_1')
            out = tf.layers.dense(tmp, 2, name='To_vector')
            loss = tf.losses.mean_squared_error(self.xs, out)
            step = tf.train.RMSPropOptimizer(0.001).minimisze(loss)
        return out, loss, step

    def decoder_training(self, size, dim):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            miu = tf.layers.dense(self.en_state, 1, name='miu')
            sig_hat = tf.layers.dense(self.en_state, 1, name='sig_hat')
            sig = tf.exp(sig_hat / 2)
            z = tf.tanh(miu + sig * tf.random_normal(1))

            # define rnn
            de_layers = [tf.nn.rnn_cell.BasicLSTMCell(dim) for _ in range(size)]
            de_layer = tf.nn.rnn_cell.MultiRNNCell(de_layers)
            output, state = tf.nn.dynamic_rnn(de_layer, z)

            # generate sequences # a total mess here
            tfd = tf.contrib.distributions
            mix = 0.3
            bimix_gauss = tfd.Mixture(
                cat=tfd.Categorical(probs=[mix, 1. - mix]),
                components=[
                    tfd.Normal(loc=-1., scale=0.1),
                    tfd.Normal(loc=+1., scale=0.5),
                ])
            xx = bimix_gauss.fit(self.xs)
            logits = bimix_gauss()
            kl_loss = tf.distributions.kl_divergence(logits, xx)
            r_loss = self.reconstruction_loss()

            #
            ks_w = tf.Variable(name='kl_loss_weight', dtype=tf.float32)
            loss = ks_w * kl_loss + r_loss

            #
            step = tf.train.AdamOptimizer().minimize(loss)
        return logits, loss, step

    def reconstruction_loss(self):
        return 0
