# This is the implemented and tuned model of the tutorial code from tensorflow
import tensorflow as tf


class cnn_rnn_classifier:
    def __init__(self, estimator=False, params=None, inputs=None, batch_size=None, max_len = None,
                 num_classes=None, train_mode=None, lr=0.001, optimizer='adam',
                 num_rnn_layers=3, num_rnn_nodes=128, kind_rnn_nodes='lstm', drop_rnn=0,
                 num_cnn_layers=(48, 64, 96), kernels_cnn=(5, 5, 3), strides_cnn=(1, 1, 1),
                 bn_cnn=False, drop_cnn=0):
        if estimator:
            self.params = params
            self.xs, self.ys, self.sequence_length, mode = inputs
            self.xs = tf.reshape(self.xs, [self.params.batch_size, -1, 3])
            self.mode = mode
            self.params.add_hparam("train_mode", mode == tf.estimator.ModeKeys.TRAIN)
        else:
            self.params = tf.contrib.training.HParams(
                batch_size=batch_size, max_len=max_len, num_class=num_classes,
                train_mode=train_mode, lr=lr, optimizer=optimizer,
                num_r_l=num_rnn_layers, num_r_n=num_rnn_nodes, rnn_node=kind_rnn_nodes, dr_rnn=drop_rnn,
                num_c_l=num_cnn_layers, ker_cnn=kernels_cnn, str_cnn=strides_cnn,
                bn_cnn=bn_cnn, dr_cnn=drop_cnn
            )
            with tf.name_scope('placeholder'):
                self.xs = tf.placeholder(dtype=tf.float32, shape=(batch_size, max_len, num_rnn_layers))
                self.ys = tf.placeholder(dtype=tf.int32, shape=batch_size)
                self.sequence_length = tf.placeholder(dtype=tf.int32, shape=batch_size)

        with tf.variable_scope('cnn', reuse=tf.AUTO_REUSE):
            self.cnn_out = self.init_cnn()
        with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
            self.rnn_out = self.init_rnn()
        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
            self.logits = self.init_logits()
            self.loss, self.acc, self.prediction, self.step = self.train()

        if not estimator:
            tf.summary.scalar('train_loss', self.loss)
            tf.summary.scalar('train_accuracy', self.acc)
            self.sum = tf.summary.merge_all()

            self.saver = tf.train.Saver()

            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.sess.run(tf.global_variables_initializer())

    def init_cnn(self):
        n_layers, kernels, strides = self.params.num_c_l, self.params.ker_cnn, self.params.str_cnn
        bn, dr, mode = self.params.bn_cnn, self.params.dr_cnn, self.params.train_mode
        out = self.xs
        for i in range(len(n_layers)):
            out = tf.layers.conv1d(out, n_layers[i], kernels[i], strides[i], padding='same',
                                   activation=None, name='conv_{}'.format(i))
            if bn:
                out = tf.layers.batch_normalization(out, training=mode)
            if dr > 0:
                out = tf.layers.dropout(out, dr, training=mode)
        return out

    def init_rnn(self):
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
            fw = [tf.nn.rnn_cell.DropoutWrapper(c, output_keep_prob=1-dr) for c in fw]
            bw = [tf.nn.rnn_cell.DropoutWrapper(c, output_keep_prob=1-dr) for c in bw]

        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            fw, bw, inputs=self.cnn_out, sequence_length=self.sequence_length, dtype=tf.float32
        )
        return outputs

    def init_logits(self):
        # choose the output according to the real length and hidden states
        # outputs is [batch_size, L, N] where L is the maximal sequence length and N
        # the number of nodes in the last layer.
        mask = tf.tile(
            tf.expand_dims(tf.sequence_mask(self.sequence_length, tf.shape(self.rnn_out)[1]), 2),
            [1, 1, tf.shape(self.rnn_out)[2]])
        zero_outside = tf.where(mask, self.rnn_out, tf.zeros_like(self.rnn_out))
        real_out = tf.reduce_sum(zero_outside, axis=1)

        # add dense layer
        return tf.layers.dense(real_out, self.params.num_class)

    def train(self, lr_decay=None, epoch=None):
        # calculate loss
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ys, logits=self.logits)
        loss = tf.reduce_mean(losses)
        # calculate accuracy
        pred = tf.argmax(self.logits, axis=1)
        acc = tf.metrics.accuracy(self.ys, pred)
        # define optimizer and traininf step
        opt_name, lr = self.params.optimizer, self.params.lr
        if lr_decay:
            if not epoch:
                raise ValueError('Number of Epoch is not defined')
            lr /= (1 + lr_decay * epoch)

        step = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=lr,
            optimizer="opt_name",
            summaries=["learning_rate", "loss", "gradients", "gradient_norm"])
        return loss, acc, pred, step
