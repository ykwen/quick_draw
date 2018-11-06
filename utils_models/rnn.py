import tensorflow as tf
import functools
from utils_models.utils import bivariate_normal, sample_fn


# an basic rnn model for implementing
class basic_rnn:
    def __init__(self, params, estimator=False, xs=None, ys=None, lens=None):
        self.params = params
        if estimator:
            self.xs, self.ys, self.seq_len = xs, ys, lens
        else:
            # if it's not used in estimator, need to define placeholder
            with tf.name_scope("placeholder"):
                self.xs = tf.placeholder(tf.int32,
                                         [self.params.batch_size, self.params.max_len, self.params.one_input_shape],
                                         "inputs")
                self.ys = tf.placeholder(tf.int32, [self.params.batch_size, None], "classes")
                self.seq_len = tf.placeholder(tf.int32, self.params.batch_size, "sequence_length")

        self.rnn_out = self.get_rnn_layers()
        self.logits, self.loss = self.get_loss()

        self.step = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=tf.train.get_global_step(),
            learning_rate=self.params.lr,
            optimizer=self.params.opt_name,
            summaries=["learning_rate", "loss", "gradients", "gradient_norm"])

        if not estimator:
            # define summary and calculate result
            if self.params.classifier:
                self.preds = tf.argmax(self.logits, axis=1)
                self.acc = tf.metrics.accuracy(self.ys, self.preds)
                tf.summary.scalar('train_accuracy', self.acc)
            self.sum = tf.summary.merge_all()

            self.saver = tf.train.Saver()

            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.sess.run(tf.global_variables_initializer())

            if self.params.restore:
                restore = tf.train.Saver()
                restore.restore(self.sess, self.params.model)

    def get_rnn_layers(self):
        '''
        Using model self.params
        :return: certain type of rnn cell
        '''
        def _get_rnn_cell():
            node = self.params.rnn_node
            if node == 'lstm':
                cell = tf.nn.rnn_cell.BasicLSTMCell
            elif node == 'gru':
                cell = tf.nn.rnn_cell.GRUCell
            else:
                cell = tf.nn.rnn_cell.BasicRNNCell
            return cell

        # get rnn cell type
        cell = _get_rnn_cell()

        if self.params.bidir:
            return self.get_bi_rnn(cell)
        else:
            return self.get_basic_rnn(cell)

    def get_bi_rnn(self, cell):
        '''
        :param cell: a certain type of rnn cell
        :return: final output of bi-direction dynamic rnn according to certain sequence length
        '''
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
        real_out = tf.reshape(tf.gather(outputs, self.seq_len, axis=1), [outputs.shape[0], outputs.shape[-1]])
        return real_out

    def get_basic_rnn(self, cell):
        '''
        :param cell: a certain type of rnn cell
        :return: output of length of single direction rnn cell
        '''
        layers = [cell(self.params.num_r_n) for _ in range(self.params.num_r_l)]
        if self.params.dr > 0:
            layers = [tf.nn.rnn_cell.DropoutWrapper(c, output_keep_prob=1 - self.params.dr) for c in layers]

        cells = tf.nn.rnn_cell.MultiRNNCell(layers)

        outputs, _ = tf.nn.dynamic_rnn(cells, self.xs, sequence_length=self.seq_len)
        return tf.reshape(tf.gather(outputs, self.seq_len, axis=1), [outputs.shape[0], outputs.shape[-1]])

    def get_loss(self):
        '''
        To be implemented in special cases
        :return: loss
        '''
        raise NotImplementedError("The loss function is not implemented")


class rnn_encoder(basic_rnn):
    def get_loss(self):
        '''

        :return: classifier logits output and softmax cross entropy error
        '''
        with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
            logits = tf.layers.dense(self.rnn_out, self.params.num_classes)
            loss = tf.losses.sparse_softmax_cross_entropy(self.ys, logits)
        return logits, loss


class rnn_decoder(basic_rnn):
    def get_basic_rnn(self, cell):
        '''

        :param cell: a certain type of rnn cell
        :return: decoder outputs
        '''
        de_layers = [tf.nn.rnn_cell.BasicLSTMCell(self.params.num_r_n) for _ in range(self.params.num_r_l)]
        de_cells = tf.nn.rnn_cell.MultiRNNCell(de_layers)

        def _get_proj_layer():
            '''

            :return: projection to parameters
            '''
            return functools.partial(
                tf.layers.dense, units=6 * self.params.gmm_dim + 3
            )

        def _train_with_inputs_helper():
            '''

            :return: training helper with given data and types
            '''
            s0 = tf.constant([[0, 0, 1, 0, 0]], dtype=tf.float32)
            s0s = tf.reshape(tf.tile(s0, self.rnn_out.shape[0]), [self.rnn_out.shape[0], 1, s0.shape[0]])
            s_ = tf.concat([s0s, self.xs], axis=1)
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=s_,
                sequence_length=self.params.max_len + 1,
                time_major=False
            )
            return helper

        def _train_self_helper():
            '''

            :return: self-defined helper take only initial inputs
            '''
            def _sample_fn(de_out):
                '''
                implementing in tf helper.py
                :param de_out: decoder raw outputs
                :return: [N, (x, y, p1, p2, p3)]
                '''
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
                '''
                generate p3 = 1
                :param inputs: decoder transformed output
                :return: N boolean of whether its ended
                '''
                n = self.params.batch_size
                return tf.equal(tf.slice(inputs, [0, 4], [n, 1]), tf.constant(1, tf.int32))

            s0 = tf.constant([[0, 0, 1, 0, 0]], dtype=tf.float32)
            helper = tf.contrib.seq2seq.InferenceHelper(
                sample_fn=_sample_fn,
                sample_shape=[self.params.input_shape[0], self.params.input_shape[2]],
                sample_dtype=self.params.input_dtype,
                start_inputs=s0,
                end_fn=_end_fn
            )
            return helper
        if self.params.train_with_inputs and self.params.mode == tf.estimator.ModeKeys.TRAIN:
            helper = _train_with_inputs_helper()
        else:
            helper = _train_self_helper()

        # output should be [N, 5M+M+3]
        decoder = tf.contrib.seq2seq.Decoder(cell=de_cells,
                                             helper=helper,
                                             output_layer=_get_proj_layer())
        out, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                      output_time_major=False,
                                                      impute_finished=True,
                                                      maximum_iterations=self.params.max_len + 1)
        if self.params.mode == tf.estimator.ModeKeys.TRAIN:
            # to calculate losses
            mask = tf.tile(
                tf.expand_dims(tf.sequence_mask(self.seq_len + 1, out.shape[1]), 2),
                [1, 1, out.shape[2]])
            return tf.where(mask, out, tf.zeros_like(out))
        return out

    def get_loss(self):
        '''
        the logits from loss is the raw output of rnn. To transform, call function sample_fn
        :return:
        '''
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            logits, loss = self.cal_decoder_loss(self.rnn_out)
        return sample_fn(self.params, logits), loss

    def cal_decoder_loss(self, de_out):
        '''

        :param de_out: parameters
        :return: loss = reconstruction loss + kl loss * kl weight
        '''
        n, m = self.params.batch_size, self.params.gmm_dim
        # get parameters for normal distribution from outputs
        muxs = tf.slice(de_out, [0, 0], [n, m])
        muys = tf.slice(de_out, [0, m], [n, m])
        sigxs_ = tf.slice(de_out, [0, 2 * m], [n, m])
        sigys_ = tf.slice(de_out, [0, 3 * m], [n, m])
        cors_ = tf.slice(de_out, [0, 4 * m], [n, m])
        weights_ = tf.slice(de_out, [0, 5 * m], [n, m])
        qs_ = tf.slice(de_out, [0, 6 * m], [n, 3])

        # add randomness
        if self.params.mode != tf.estimator.ModeKeys.TRAIN and self.params.temper:
            weights_, qs_ = weights_ / self.params.temper, qs_ / self.params.temper

        weights = tf.nn.softmax(weights_)
        sigxs, sigys, cors = tf.nn.softmax(sigxs_), tf.nn.softmax(sigys_), tf.nn.tanh(cors_)
        if self.params.mode != tf.estimator.ModeKeys.TRAIN and self.params.temper:
            t = tf.sqrt(self.params.temper)
            sigxs, sigys = sigxs * t, sigys * t

        loss_params = [muxs, muys, sigxs, sigys, cors, weights, qs_]
        L_R = self.reconstruction_loss(loss_params)
        L_kl = self.KL_loss(sigxs, sigys, muxs, muys)
        kl_w = tf.get_variable(name="kl_loss_weight", shape=[1], dtype=tf.float32)
        return loss_params, tf.add(L_R, tf.multiply(L_kl, kl_w))

    def reconstruction_loss(self, loss_params):
        '''

        :param loss_params:
        :return: loss = ls + lp
        '''
        qs = loss_params[-1]
        N_max = tf.reduce_sum(self.seq_len + self.xs.shape[0])

        # Ls, inputs is (N, L, 5), choose the first 2 column
        deltas = tf.slice(self.xs, [0, 0, 0], [self.xs.shape[0], self.seq_len, 2])
        norm_sum = bivariate_normal(deltas, loss_params)
        Ls = (- 1 / N_max) * tf.reduce_sum(tf.log(norm_sum))

        # Lp
        ps = tf.reshape(tf.slice(self.xs, [0, 0, 2], [self.xs.shape[0], self.params.max_len, 3]), [-1, 3])
        qs = tf.reshape(qs, [-1, 3])
        Lp = (- 1 / N_max) * tf.reduce_sum(tf.losses.softmax_cross_entropy(onehot_labels=ps, logits=qs))

        return tf.add(Ls, Lp)

    def KL_loss(self, sigxs, sigys, muxs, muys):
        lx = -tf.divide(tf.reduce_mean(tf.subtract(tf.add(1, sigxs), tf.add(tf.square(muxs), sigxs))), 2)
        ly = -tf.divide(tf.reduce_mean(tf.subtract(tf.add(1, sigys), tf.add(tf.square(muys), sigys))), 2)
        return tf.add(lx, ly)

