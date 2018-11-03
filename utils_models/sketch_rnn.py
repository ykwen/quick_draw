import tensorflow as tf


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
            self.loss, self.step, self.sketch_out = self.decoder_training()
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
        def _GMM(inputs):
            meanx = tf.get_variable(name="meanx", shape=self.params.gmm_dim, dtype=tf.float64)
            meany = tf.get_variable(name="meany", shape=self.params.gmm_dim, dtype=tf.float64)
            scalex = tf.get_variable(name="scalex", shape=self.params.gmm_dim, dtype=tf.float64)
            scaley = tf.get_variable(name="scaley", shape=self.params.gmm_dim, dtype=tf.float64)
            correlation = tf.get_variable(name="correlation", shape=self.params.gmm_dim, dtype=tf.float64)
            gmm_biv = tf.contrib.distribution.MultivariateNormalDiag()
            gmm_out = gmm_biv.prob(inputs).eval()
            pxy = tf.layers.dense(gmm_out, self.params.gmm_dim, name="gmm_weight")
            p_p = tf.layers.dense(gmm_out, 3, name="gmm_weight")


        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            miu = tf.layers.dense(self.en_out, self.en_out.shape[1], name='miu')
            sig_hat = tf.layers.dense(self.en_out, self.en_out.shape[1], name='sig_hat')
            sig = tf.exp(sig_hat / 2)
            z = tf.add(miu, tf.matmul(sig, tf.random_normal(self.en_out.shape[1])))

            h0 = tf.reshape(tf.tanh(tf.layers.dense(z, self.en_out.shape[1] * 2)), [self.en_out.shape[0], self.en_out.shape[1], 2])
            s0 = tf.constant([0, 0, 1, 0, 0], dtype=tf.float32)

            # define rnn decoder
            de_layers = [tf.nn.rnn_cell.BasicLSTMCell(self.params.decoder_size) for _ in range(self.params.decoder_dim)]
            de_cells = tf.nn.rnn_cell.MultiRNNCell(de_layers)
            helper = tf.contrib.seq2seq.TrainingHelper(
                input=s0,
                sequence_length=self.params.max_len,
                time_major=False
            )
            decoder = tf.contrib.seq2seq.Decoder(cell=de_cells, helper=helper,
                                                 initial_state=h0, output_layer=tf.layers.dense)
            out, _, out_seqlen = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                   output_time_major=False,
                                                                   impute_finished=True,
                                                                   maximum_iterations=20)
            de_out = tf.gather(out, out_seqlen, axis=1)

            # applying GMM to decoder output
            de_out_sketch = _GMM(de_out)

            loss, step = _get_decoder_step(de_out_sketch)
        return loss, step, de_out_sketch

    def reconstruction_loss(self):
        return 0
