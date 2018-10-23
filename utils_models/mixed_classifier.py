# This is the implemented and tuned model of the tutorial code from tensorflow
import tensorflow as tf


class rnn_cnn_classifier:
    def __init__(self, input_shape, num_classes, mode,
                 num_rnn_layers=3, num_rnn_nodes=128, kind_rnn_nodes='lstm',
                 bn_rnn=False, drop_rnn=0, lr_rnn=0.001, optimizer_rnn='adam',
                 num_cnn_layers=[48, 64, 96], kernels_cnn=[5, 5, 3], strides_cnn=[1, 1, 1],
                 bn_cnn=False, drop_cnn=0, lr_cnn=0.001, optimizer_cnn='dadm'):
        self.params = tf.contrib.training.HParams(
            batch_shape=input_shape, num_class=num_classes, train_mode=mode,
            num_r_l=num_rnn_layers, num_r_n=num_rnn_nodes, rnn_node=kind_rnn_nodes,
            bn_rnn=bn_rnn, dr_rnn=drop_rnn, lr_rnn=lr_rnn, op_rnn=optimizer_rnn,
            num_c_l=num_cnn_layers, ker_cnn=kernels_cnn, str_cnn=strides_cnn,
            bn_cnn=bn_cnn, dr_cnn=drop_cnn, lr_cnn=lr_cnn, op_cnn=optimizer_cnn
        )

        with tf.name_scope('placeholder'):
            self.xs = tf.placeholder(dtype=tf.float32, shape=input_shape)
            self.ys = tf.placeholder(dtype=tf.float32, shape=input_shape[0])
            self.sequence_length = tf.placeholder(dtype=tf.int16, shape=input_shape[0])

        with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
            self.rnn_out = self.init_rnn()
        with tf.variable_scope('cnn', reuse=tf.AUTO_REUSE):
            self.cnn_out = self.init_cnn()
        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
            self.logits = self.init_logits()
            self.loss, self.acc, self.step = self.train()

        tf.summary.scalar('train_loss', self.loss)
        tf.summary.scalar('train_accuracy', self.acc)
        self.sum = tf.summary.merge_all()

        self.saver = tf.train.Saver()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer)

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
            fw, bw, inputs=self.xs, sequence_length=self.sequence_length, dtype=tf.float32
        )
        return outputs

    def init_cnn(self):
        return None
