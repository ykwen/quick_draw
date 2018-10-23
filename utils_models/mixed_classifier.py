# This is the implemented and tuned model of the tutorial code from tensorflow
import tensorflow as tf


class rnn_cnn_classifier:
    def __init__(self, input_shape, num_classes, mode, lr=0.001, optimizer='adam',
                 num_rnn_layers=3, num_rnn_nodes=128, kind_rnn_nodes='lstm', drop_rnn=0,
                 num_cnn_layers=(48, 64, 96), kernels_cnn=(5, 5, 3), strides_cnn=(1, 1, 1),
                 bn_cnn=False, drop_cnn=0):
        self.params = tf.contrib.training.HParams(
            batch_shape=input_shape, num_class=num_classes, train_mode=mode, lr=lr, optimizer=optimizer,
            num_r_l=num_rnn_layers, num_r_n=num_rnn_nodes, rnn_node=kind_rnn_nodes, dr_rnn=drop_rnn,
            num_c_l=num_cnn_layers, ker_cnn=kernels_cnn, str_cnn=strides_cnn,
            bn_cnn=bn_cnn, dr_cnn=drop_cnn
        )

        with tf.name_scope('placeholder'):
            self.xs = tf.placeholder(dtype=tf.float32, shape=input_shape)
            self.ys = tf.placeholder(dtype=tf.float32, shape=input_shape[0])
            self.sequence_length = tf.placeholder(dtype=tf.int16, shape=input_shape[0])

        with tf.variable_scope('cnn', reuse=tf.AUTO_REUSE):
            self.cnn_out = self.init_cnn()
        with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
            self.rnn_out = self.init_rnn()
        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
            self.logits = self.init_logits()
            self.loss, self.acc, self.step = self.train()

        tf.summary.scalar('train_loss', self.loss)
        tf.summary.scalar('train_accuracy', self.acc)
        self.sum = tf.summary.merge_all()

        self.saver = tf.train.Saver()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer)

    def init_cnn(self):
        n_layers, kernels, strides = self.params.num_c_l, self.params.ker_cnn, self.params.str_cnn
        bn, dr, mode = self.params.bn_cnn, self.params.dr_cnn, self.params.train_mode
        out = self.xs
        for i in range(len(n_layers)):
            if bn:
                out = tf.layers.batch_normalization(bn, training=mode)
            if dr > 0:
                out = tf.layers.dropout(out, dr, training=mode)
            out = tf.layers.conv1d(out, n_layers[i], kernels[i], strides[i], padding='same',
                                   activation=None, name='conv_{}'.format(i))
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
            fw = [tf.contrib.rnn.DropoutWrapper(c) for c in fw]
            bw = [tf.contrib.rnn.DropoutWrapper(c) for c in bw]

        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            fw, bw, inputs=self.cnn_out, sequence_length=self.sequence_length, dtype=tf.float32
        )
        return outputs


