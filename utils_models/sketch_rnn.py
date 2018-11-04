import tensorflow as tf
from utils_models.utils import bivariate_normal


class sketch_rnn:
    def __init__(self, estimator=False, params=None,
                 inputs=None, batch_size=None, learning_rate=None, optimizer=None,
                 num_classes=None, max_len=None, encoder_dim=None,
                 encoder_restore=False, classifier=True,
                 encoder_path=None, decoder=False, vector_shape=None,
                 decoder_size=None, decoder_dim=None, gmm_dim=None):
        if estimator:
            self.params = params
            self.xs, self.ys, self.seq_len, self.mode = inputs
            self.params.add_hparam("train_mode", self.mode == tf.estimator.ModeKeys.TRAIN)
        else:
            self.en_size = max_len
            self.en_dim = encoder_dim
            self.batch_size = batch_size
            self.num_classes = num_classes

            with tf.name_scope('placeholder'):
                self.xs = tf.placeholder([batch_size, max_len, 5], tf.float32)
                self.ys = tf.placeholder(self.batch_size, tf.int8)
                if decoder and not vector_shape:
                    self.vecs = tf.placeholder(vector_shape)

        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            self.en_out = self.get_rnn_layers()

        if encoder_restore:
            self.restorer = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                      scope='encoder'))
            self.restorer.restore(self.sess, encoder_path)

        if classifier:
            # for classification layer
            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                self.output, self.loss, self.acc, self.step = self.get_logits()
        elif decoder:
            # for training decoder
            self.loss, self.step, self.de_out = self.decoder_training()
        else:
            raise ValueError("Invalid Mode")

        if not estimator:
            tf.summary.scalar('train_loss', self.loss)
            tf.summary.scalar('train_accuracy', self.acc)
            self.sum = tf.summary.merge_all()

            self.saver = tf.train.Saver()

            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.sess.run(tf.global_variables_initializer())

    def get_rnn_layers(self):
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

    def get_logits(self):
        logits = tf.layers.dense(self.en_out, self.params.num_classes, name='classify')
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ys, logits=logits)
        )
        acc = tf.metrics.accuracy(self.ys, logits)
        step = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=self.params.lr,
            optimizer=self.params.opt_name,
            summaries=["learning_rate", "loss", "gradients", "gradient_norm"])
        return logits, loss, acc, step

    def decoder_training(self):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            # (N, D)
            miu = tf.layers.dense(self.en_out, self.en_out.shape[1], name='miu')
            sig_hat = tf.layers.dense(self.en_out, self.en_out.shape[1], name='sig_hat')
            sig = tf.exp(sig_hat / 2)
            z = tf.add(miu, tf.matmul(sig, tf.random_normal(self.en_out.shape[1])))

            # [h0; c0] (N, 2, D)
            h0 = tf.reshape(tf.tanh(tf.layers.dense(z, self.en_out.shape[1] * 2, name="Wz+b")),
                            [self.en_out.shape[0], 2, self.en_out.shape[1]])
            s0 = tf.constant([[0, 0, 1, 0, 0]], dtype=tf.float32)

            # concatenate z and points for decoder input
            # inputs (N, L, 5), z (N, D), s0 (5)
            zs = tf.reshape(tf.tile(z, [1, self.params.max_len]),
                            [self.en_out.shape[0], self.params.max_len, self.en_out.shape[1]])
            s0s = tf.reshape(tf.tile(s0, self.en_out.shape[0]), [self.en_out.shape[0], 1, s0.shape[0]])
            s_z = tf.concat([tf.concat([s0s, self.xs], axis=1), zs], axis=2)

            # define rnn decoder
            de_layers = [tf.nn.rnn_cell.BasicLSTMCell(self.params.decoder_size) for _ in range(self.params.decoder_dim)]
            de_cells = tf.nn.rnn_cell.MultiRNNCell(de_layers)
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=s_z,
                sequence_length=self.params.max_len + 1,
                time_major=False
            )
            decoder = tf.contrib.seq2seq.Decoder(cell=de_cells, helper=helper,
                                                 initial_state=h0)
            out, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                          output_time_major=False,
                                                          impute_finished=True,
                                                          maximum_iterations=self.params.max_len + 1)
            mask = tf.tile(
                tf.expand_dims(tf.sequence_mask(self.seq_len + 1, out.shape[1]), 2),
                [1, 1, out.shape[2]])
            de_out = tf.where(mask, out, tf.zeros_like(out))

            # applying GMM to decoder output
            loss_params, loss = self.get_decode_loss(de_out)

            step = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                learning_rate=self.params.lr,
                optimizer=self.params.opt_name,
                summaries=["learning_rate", "loss", "gradients", "gradient_norm"])
        return loss, step, loss_params

    def get_decode_loss(self, de_out):
        muxs = tf.layers.dense(de_out, self.params.gmm_dim, name="mux")
        muys = tf.layers.dense(de_out, self.params.gmm_dim, name="muy")
        sigxs_ = tf.layers.dense(de_out, self.params.gmm_dim, name="sigx")
        sigys_ = tf.layers.dense(de_out, self.params.gmm_dim, name="sigy")
        cors_ = tf.layers.dense(de_out, self.params.gmm_dim, name="cors")
        weights_ = tf.layers.dense(de_out, self.params.gmm_dim, name="gmm_w")
        qs_ = tf.layers.dense(de_out, 3, name="q(pi)")

        # add randomness
        if self.params.temper:
            weights_, qs_ = weights_ / self.params.temper, qs_ / self.params.temper

        weights = tf.nn.softmax(weights_)
        sigxs, sigys, cors = tf.nn.softmax(sigxs_), tf.nn.softmax(sigys_), tf.nn.tanh(cors_)
        if self.params.temper:
            t = tf.sqrt(self.params.temper)
            sigxs, sigys = sigxs * t, sigys * t

        loss_params = [muxs, muys, sigxs, sigys, cors, weights, qs_]
        L_R = self.reconstruction_loss(loss_params)
        L_kl = self.KL_loss(sigxs, sigys, muxs, muys)
        kl_w = tf.get_variable(name="kl_loss_weight", shape=[1], dtype=tf.float32)
        return loss_params, tf.add(L_R, tf.multiply(L_kl, kl_w))

    def reconstruction_loss(self, loss_params):
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
