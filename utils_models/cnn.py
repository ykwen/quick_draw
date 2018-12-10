import tensorflow as tf
from tensorflow.layers import batch_normalization as bn
from tensorflow.layers import dropout as dr


class ResNet_18:
    def __init__(self, batch_size, num_class, params):
        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.params, self.n = params, batch_size

        # define place holders
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(shape=[batch_size, 28, 28, 1], dtype=tf.float32)
            self.ys = tf.placeholder(shape=[batch_size, ], dtype=tf.int64)

        # define layers
        with tf.variable_scope('layers', reuse=tf.AUTO_REUSE):
            # conv->maxpool->conv*2->conv*2->conv*2->conv*2>GAP>output
            self.conv1 = tf.layers.conv2d(self.xs, filters=32, kernel_size=3, strides=1, padding='same', name='conv1')
            self.max1 = tf.layers.max_pooling2d(self.conv1, pool_size=2, strides=2, padding='same', name='max1')
            self.w_, self.c_ = 14, 32
            self.conv2 = self.conv_unit(x_in=self.max1, num_conv=32, ker_size=3, index=2, begin_new=False)
            self.conv2 = self.conv_unit(x_in=self.conv2, num_conv=32, ker_size=3, index=3, begin_new=False)
            self.conv3 = self.conv_unit(self.conv2, 64, 3, 4, True)
            self.conv3 = self.conv_unit(self.conv3, 64, 3, 5, False)
            self.conv4 = self.conv_unit(self.conv3, 128, 3, 6, True, True)
            self.conv4 = self.conv_unit(self.conv4, 128, 3, 7, False)
            self.conv5 = self.conv_unit(self.conv4, 256, 3, 8, True)
            self.conv5 = self.conv_unit(self.conv5, 256, 3, 9, False)

            self.glo_ave_pool = tf.reduce_mean(self.conv5, axis=[1, 2])

            self.final_out = tf.layers.dense(self.glo_ave_pool, 200, activation=tf.nn.relu, name='FC')
            self.logits = tf.layers.dense(self.final_out, num_class, name='logits')

        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            # define loss and optimization methods and output fot test
            self.labels = tf.one_hot(self.ys, num_class)
            self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits,
                                                                     name='cross_entropy_loss')
            self.loss = tf.reduce_sum(self.losses)

        self.global_step = tf.train.get_or_create_global_step()

        self.step = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=self.global_step,
            learning_rate=self.params.lr,
            optimizer=self.params.opt_name,
            clip_gradients=self.params.clip_gradients,
            summaries=[]
        )

        # define summary and calculate result
        tf.summary.scalar('train_loss', self.loss)
        if self.params.classifier:
            self.preds = tf.argmax(self.logits, axis=1)
            self.acc = tf.contrib.metrics.accuracy(labels=self.ys,
                                                   predictions=self.preds)
            tf.summary.scalar('train_accuracy', self.acc)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        if self.params.mode == tf.estimator.ModeKeys.TRAIN:
            self.sum = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.params.summary)
            self.summary_writer.add_graph(self.sess.graph)

        if self.params.restore:
            restore = tf.train.Saver()
            restore.restore(self.sess, self.params.model)

    def conv_unit(self, x_in, num_conv, ker_size, index, begin_new, test=False):
        if begin_new:
            s_begin = 2
        else:
            s_begin = 1
        out = tf.layers.conv2d(x_in, num_conv, ker_size, s_begin, padding='same', activation=tf.nn.relu,
                               name='conv' + str(index) + '_1_1')
        out = tf.layers.conv2d(out, num_conv, ker_size, 1, padding='same', activation=None,
                               name='conv' + str(index) + '_1_2')

        x_out = self.projection(x_in, begin_new, num_conv, index)

        return tf.nn.relu(out + x_out)

    def projection(self, inputs, begin_new, num_conv, index):
        if begin_new:
            w_ = int(self.w_ / 2) + int(self.w_ % 2)
            shapes = [self.w_ * self.w_ * self.c_, w_ * w_ * num_conv]
            self.w_ = w_
            self.c_ = num_conv
        else:
            shapes = [self.w_ * self.w_ * self.c_, self.w_ * self.w_ * self.c_]
        in_ = tf.reshape(inputs, [self.n, -1])
        weight = tf.get_variable(name='conv' + str(index) + '_proj',
                                 shape=shapes, dtype=tf.float32)
        return tf.reshape(tf.matmul(in_, weight), [self.n, self.w_, self.w_, num_conv])

