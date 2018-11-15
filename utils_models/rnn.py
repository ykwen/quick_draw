import tensorflow as tf
import functools
import math
from tensorflow.python.layers.core import Dense
from utils_models.utils import sample_points


# an basic rnn model for implementing
class RNNBasic:
    def __init__(self, params, estimator=False, xs=None, ys=None, lens=None, decoder=False):
        self.params = params
        self.pi = math.pi
        if estimator:
            self.xs, self.ys, self.seq_len = xs, ys, lens
        else:
            # if it's not used in estimator, need to define placeholder
            with tf.name_scope("placeholder"):
                self.xs = tf.placeholder(tf.float32,
                                         [self.params.batch_size, self.params.max_len, self.params.one_input_shape],
                                         "inputs")
                if decoder:
                    self.ys = tf.placeholder(
                        tf.float32,
                        [self.params.batch_size, self.params.max_len, self.params.one_input_shape],
                        "shifted_input")
                else:
                    self.ys = tf.placeholder(tf.int64, self.params.batch_size, "classes")
                self.seq_len = tf.placeholder(tf.int32, self.params.batch_size, "sequence_length")

        self.rnn_out, self.rnn_out_all = self.get_rnn_layers()
        self.logits, self.loss = self.get_loss()

        self.step = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=tf.train.get_global_step(),
            learning_rate=self.params.lr,
            optimizer=self.params.opt_name,
            clip_gradients=self.params.clip_gradients,
            summaries=["learning_rate"]
        )

        if not estimator:
            # define summary and calculate result
            if self.params.classifier:
                self.preds = tf.argmax(self.logits, axis=1)
                self.acc = tf.contrib.metrics.accuracy(labels=self.ys,
                                                       predictions=self.preds)
                tf.summary.scalar('train_accuracy', self.acc)

            tf.summary.scalar('train_loss', self.loss)

            self.saver = tf.train.Saver()

            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.sess.run(tf.global_variables_initializer())

            self.sum = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.params.summary)
            self.summary_writer.add_graph(self.sess.graph)

            if self.params.restore:
                restore = tf.train.Saver()
                restore.restore(self.sess, self.params.model)

    def get_rnn_layers(self):
        """
        Using model self.params
        :return: certain type of rnn cell
        """
        def _get_rnn_cell():
            node = self.params.rnn_node
            if node == 'lstm':
                cell = tf.nn.rnn_cell.BasicLSTMCell
            elif node == 'gru':
                cell = tf.nn.rnn_cell.GRUCell
            else:
                cell = tf.nn.rnn_cell.BasicRNNCell
            return functools.partial(cell, activation=self.params.activation)

        # get rnn cell type
        cell = _get_rnn_cell()

        if self.params.bidir:
            return self.get_bi_rnn(cell)
        else:
            return self.get_basic_rnn(cell)

    def get_bi_rnn(self, cell):
        """
        :param cell: a certain type of rnn cell
        :return: final output of bi-direction dynamic rnn according to certain sequence length
        """
        # define forward and backward layers
        fw = [cell(self.params.num_r_n) for _ in range(self.params.num_r_l)]
        bw = [cell(self.params.num_r_n) for _ in range(self.params.num_r_l)]

        # add drop out according to drop out rate
        dr = self.params.dr_rnn
        if dr > 0:
            fw = [tf.nn.rnn_cell.DropoutWrapper(c, output_keep_prob=1 - dr) for c in fw]
            bw = [tf.nn.rnn_cell.DropoutWrapper(c, output_keep_prob=1 - dr) for c in bw]

        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            fw, bw, inputs=self.xs, sequence_length=self.seq_len, dtype=tf.float32
        )
        # use the final hidden states, output is [N, max_length, D]
        real_out = tf.reshape(
            tf.gather_nd(outputs,
                         tf.concat([
                             tf.reshape(tf.range(self.params.batch_size), [self.params.batch_size, 1]),
                             tf.reshape(tf.subtract(self.seq_len, 1), [self.params.batch_size, 1])], axis=1)),
            [outputs.shape[0], outputs.shape[-1]])
        return real_out, outputs

    def get_basic_rnn(self, cell):
        """
        return output of single direction rnn
        :param cell: a certain type of rnn cell
        :return: output of length of single direction rnn cell
        """
        layers = [cell(self.params.num_r_n) for _ in range(self.params.num_r_l)]
        if self.params.dr_rnn > 0:
            layers = [tf.nn.rnn_cell.DropoutWrapper(c, output_keep_prob=1 - self.params.dr_rnn) for c in layers]

        cells = tf.nn.rnn_cell.MultiRNNCell(layers)

        outputs, _ = tf.nn.dynamic_rnn(cells, self.xs, sequence_length=self.seq_len, dtype=tf.float32)
        real_output = tf.reshape(
            tf.gather_nd(outputs,
                         tf.concat([
                             tf.reshape(tf.range(self.params.batch_size), [self.params.batch_size, 1]),
                             tf.reshape(tf.subtract(self.seq_len, 1), [self.params.batch_size, 1])], axis=1)),
            [outputs.shape[0], outputs.shape[-1]])
        return real_output, outputs

    def get_loss(self):
        """
        To be implemented in special cases
        :return: loss
        """
        raise NotImplementedError("The loss function is not implemented")


class RNNEncoder(RNNBasic):
    def get_loss(self):
        """

        :return: classifier logits output and softmax cross entropy error
        """
        with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
            logits = tf.layers.dense(self.rnn_out, self.params.num_classes)
            losses = tf.losses.sparse_softmax_cross_entropy(self.ys, logits)
            loss = tf.reduce_sum(losses)
        return logits, loss


class RNNDecoder(RNNBasic):
    def get_bi_rnn(self, cell):
        raise ValueError("Cannot use bidirection RNN for decoder")

    def get_basic_rnn(self, cell):
        """

        :param cell: a certain type of rnn cell
        :return: decoder outputs
        """
        if self.params.num_r_l > 1:
            de_layers = [tf.nn.rnn_cell.BasicLSTMCell(self.params.num_r_n) for _ in range(self.params.num_r_l)]
            de_cells = tf.nn.rnn_cell.MultiRNNCell(de_layers)
        else:
            de_cells = tf.nn.rnn_cell.BasicLSTMCell(self.params.num_r_n)

        def _train_with_inputs_helper():
            """

            :return: training helper with given data and types
            """
            train_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=self.xs,
                sequence_length=self.seq_len,
                time_major=False
            )
            return train_helper

        def _train_self_helper():
            """

            :return: self-defined helper take only initial inputs
            """
            def _sample_fn(de_out):
                """
                implementing in tf helper.py
                :param de_out: decoder raw outputs
                :return: [N, (x, y, p1, p2, p3)]
                """
                n, m = self.params.batch_size, self.params.gmm_dim
                muxs = tf.slice(de_out, [0, 0], [n, m])
                muys = tf.slice(de_out, [0, m], [n, m])
                weights_ = tf.transpose(tf.slice(de_out, [0, 5 * m], [n, m]))
                qs_ = tf.slice(de_out, [0, 6 * m], [n, 3])

                x, y = tf.trace(tf.matmul(muxs, weights_)), tf.trace(tf.matmul(muys, weights_))
                x, y = tf.reshape(x, [self.params.batch_size, 1]), tf.reshape(y, [self.params.batch_size, 1])
                q = tf.one_hot(tf.argmax(qs_, axis=1), 3)
                return tf.concat([x, y, q], axis=1)

            def _end_fn(inputs):
                """
                generate p3 = 1
                :param inputs: decoder transformed output
                :return: N boolean of whether its ended
                """
                n = self.params.batch_size
                return tf.equal(tf.slice(inputs, [0, 4], [n, 1]), tf.constant(1, tf.int32))

            s0 = tf.constant([[0, 0, 1, 0, 0]], dtype=tf.float32)
            helper = tf.contrib.seq2seq.InferenceHelper(
                sample_fn=_sample_fn,
                sample_shape=[self.params.batch_size, self.params.one_input_shape],
                sample_dtype=tf.float32,
                start_inputs=s0,
                end_fn=_end_fn
            )
            return helper

        if self.params.train_with_inputs and self.params.mode == tf.estimator.ModeKeys.TRAIN:
            helper = _train_with_inputs_helper()
        else:
            helper = _train_self_helper()

        # output is states, then be projected to [N, 5M+M+3]
        M = 6 * self.params.gmm_dim + 3
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=de_cells,
                                                  helper=helper,
                                                  initial_state=de_cells.zero_state(self.params.batch_size, tf.float32),
                                                  output_layer=Dense(M, use_bias=False)
                                                  )
        out, final_len = self.dynamic_decode(decoder=decoder,
                                             output_time_major=True,
                                             impute_finished=False,
                                             maximum_iterations=self.params.max_len)

        rnn_logits = tf.transpose(out, [1, 0, 2])

        if self.params.mode == tf.estimator.ModeKeys.TRAIN:
            # to calculate losses
            mask = tf.tile(
                tf.expand_dims(tf.sequence_mask(self.seq_len, rnn_logits.shape[1]), 2),
                [1, 1, rnn_logits.shape[2]])
            return tf.where(mask, rnn_logits, tf.zeros_like(rnn_logits)), final_len
        return rnn_logits, final_len

    def get_loss(self):
        """
        the logits = [muxs, muys, sigxs, sigys, cors, weights, qs_]
        :return: sampled [N, L, 5], loss
        """
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            logits, loss = self.cal_decoder_loss(self.rnn_out)
        return sample_points(logits), loss

    def cal_decoder_loss(self, de_out):
        """

        :param de_out: parameters
        :return: loss = reconstruction loss + kl loss * kl weight
        """
        n, l, m = self.params.batch_size, self.params.max_len, self.params.gmm_dim
        # get parameters for normal distribution from outputs
        muxs = tf.slice(de_out, [0, 0, 0], [n, l, m])
        muys = tf.slice(de_out, [0, 0, m], [n, l, m])
        sigxs_ = tf.slice(de_out, [0, 0, 2 * m], [n, l, m])
        sigys_ = tf.slice(de_out, [0, 0, 3 * m], [n, l, m])
        cors_ = tf.slice(de_out, [0, 0, 4 * m], [n, l, m])
        weights_ = tf.slice(de_out, [0, 0, 5 * m], [n, l, m])
        qs_ = tf.slice(de_out, [0, 0, 6 * m], [n, l, 3])

        # add randomness
        if self.params.mode != tf.estimator.ModeKeys.TRAIN and self.params.temper:
            weights_, qs_ = weights_ / self.params.temper, qs_ / self.params.temper

        weights = tf.nn.softmax(weights_, axis=2)
        sigxs, sigys, cors = tf.exp(sigxs_), tf.exp(sigys_), tf.nn.tanh(cors_)
        if self.params.mode != tf.estimator.ModeKeys.TRAIN and self.params.temper:
            t = tf.sqrt(self.params.temper)
            sigxs, sigys = sigxs * t, sigys * t

        loss_params = [muxs, muys, sigxs, sigys, cors, weights, qs_]
        l_r = self.reconstruction_loss(loss_params)
        l_kl = self.KL_loss(sigxs, sigys, muxs, muys)
        kl_w = self.params.w_KL
        return loss_params, tf.add(l_r, l_kl * kl_w)

    def reconstruction_loss(self, loss_params):
        """

        :param loss_params: [muxs, muys, sigxs, sigys, cors, weights, qs_]
        :return: loss = ls + lp
        """
        qs = loss_params[-1]
        N_max = tf.cast(tf.reduce_sum(self.seq_len), tf.float32)

        # Ls, ys is (N, L, 5), choose the first 2 column
        deltas = tf.slice(self.ys, [0, 0, 0], [self.params.batch_size, self.params.max_len, 2])
        log_norm = self.bivariate_normal(deltas, loss_params)

        mask = tf.tile(
            tf.expand_dims(tf.sequence_mask(self.seq_len, self.params.max_len), 2),
            [1, 1, 1])
        log_norm_zero_outside = tf.where(mask, log_norm, tf.zeros_like(log_norm))

        ls = (- 1 / N_max) * tf.reduce_sum(log_norm_zero_outside)

        # Lp
        ps = tf.slice(self.ys, [0, 0, 2], [self.params.batch_size, self.params.max_len, 3])
        mask = tf.tile(
            tf.expand_dims(tf.sequence_mask(self.seq_len, self.params.max_len), 2),
            [1, 1, 3]
        )
        masked_ps, masked_qs = tf.where(mask, ps, tf.zeros_like(ps)), tf.where(mask, qs, tf.zeros_like(qs))
        lp_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=masked_ps, logits=masked_qs)
        lp = (1 / N_max) * tf.reduce_sum(lp_loss)

        # self.Ls, self.Lp, self.loss_par, self.logn = ls, lp, loss_params, log_norm
        return tf.add(ls, lp)

    def KL_loss(self, sigxs, sigys, muxs, muys):
        lx = -tf.divide(tf.reduce_mean(tf.subtract(tf.add(1., sigxs), tf.add(tf.square(muxs), sigxs))), 2.)
        ly = -tf.divide(tf.reduce_mean(tf.subtract(tf.add(1., sigys), tf.add(tf.square(muys), sigys))), 2.)
        kl_l = lx + ly
        if self.params.mode == tf.estimator.ModeKeys.TRAIN:
            if tf.train.get_global_step():
                num_step = tf.train.get_global_step()
            else:
                num_step = 0.
            eta_step = 1. - (1. - self.params.eta_min) * (self.params.R ** num_step)
            kl_l = self.params.w_KL * eta_step * tf.maximum(kl_l, self.params.kl_min)

        # self.L_kl = kl_l
        return kl_l

    def bivariate_normal(self, inputs, loss_params):
        """
        Using given function in tensorflow is convienent, but it takes 250s for 1 step, that's too slow
        While, the gpu usage is almost 0. Rewrite with inverse and det for these 2D matrix
        :param inputs: (N, max_seq_length, 2)
        :return: log (N, L, 1)
        """
        muxs, muys, sigxs, sigys, cors, weights, _ = loss_params
        N, L = inputs.shape[0], inputs.shape[1]
        M = muxs.shape[-1]

        x, y = tf.slice(inputs, [0, 0, 0], [N, L, 1]), tf.slice(inputs, [0, 0, 1], [N, L, 1])
        x, y = tf.tile(x, [1, 1, M]), tf.tile(y, [1, 1, M])

        x, y, = tf.reshape(x, [-1, 1]), tf.reshape(y, [-1, 1])
        sigxs_1, sigys_1 = tf.reshape(sigxs, [-1, 1]), tf.reshape(sigys, [-1, 1])
        sigxs_2, sigys_2, cors_2 = tf.square(sigxs_1), tf.square(sigys_1), \
                                  tf.reshape(tf.multiply(tf.multiply(sigxs, sigys), cors), [-1, 1])

        # covs = tf.reshape(tf.concat([sigxs_2, cors_, cors_, sigys_2], axis=1), [-1, 2, 2])
        covs_det = sigxs_2 * sigys_2 - cors_2 * cors_2
        covs_inv = tf.reshape(tf.concat([sigys_2, -cors_2, -cors_2, sigxs_2], axis=1) / covs_det, [-1, 2, 2])

        # Calcualte for N*L*M values which is wrong again
        inp = tf.reshape(tf.concat([x - sigxs_1, y - sigys_1], axis=1), [-1, 1, 2])
        log_Z = tf.reshape(tf.log(2 * self.pi) + 0.5 * covs_det, [-1, 1])
        tmp_y = tf.reshape(tf.matmul(tf.matmul(inp, covs_inv), tf.transpose(inp, [0, 2, 1])), [-1, 1])

        log_p = tf.reshape(-0.5 * tmp_y - log_Z, [N, L, 1, M])

        return tf.reshape(
            tf.matmul(
                tf.reshape(log_p, [N, L, 1, M]), tf.reshape(weights, [N, L, M, 1])
            ), [N, L, 1]
        )

    def dynamic_decode(self, decoder, output_time_major, impute_finished, maximum_iterations):
        """
        The tf decoder return unknown length outputs, this will return constant size with 0s
        :param decoder: tf decoder instance
        :param output_time_major: bool
        :param impute_finished: bool
        :param maximum_iterations: the output length
        :return: output, sequence length
        """
        with tf.name_scope("dynamic_decode"):
            _, inputs, states = decoder.initialize()
            final_out, final_seq = None, tf.fill([self.params.batch_size, 1], self.params.batch_size)
            for i in range(maximum_iterations):
                outputs, states, inputs, finished = decoder.step(i, inputs, states)
                if i > 0:
                    final_out = tf.concat([final_out, tf.expand_dims(outputs.rnn_output, axis=0)], axis=0)
                else:
                    final_out = tf.expand_dims(outputs.rnn_output, axis=0)
            if output_time_major:
                return final_out, final_seq
            else:
                return tf.transpose(final_out, [1, 0, 2]), final_seq

