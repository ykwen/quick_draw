# functions for calculation
import tensorflow as tf
import tensorflow_probability as tfp
from time import time as ti
import numpy as np
import math


def bivariate_normal(inputs, params):
    """
    Using given function in tensorflow is convienent, but it takes 250s for 1 step, that's too slow
    While, the gpu usage is almost 0
    :param inputs: (N, max_seq_length, 2)
    :param params: muxs, muys, sigxs, sigys, cors, weights, qs, (N, L, M)
    :return: (N, L, 1)
    """
    muxs, muys, sigxs, sigys, cors, weights, _ = params
    N, L = inputs.shape[0], inputs.shape[1]
    M = muxs.shape[-1]

    sigxs_2, sigys_2, cors_ = sigxs ** 2, sigys ** 2, sigxs * sigys * cors
    covs = tf.reshape(tf.concat([sigxs_2, cors_, cors_, sigys_2], axis=2), [N, L, M, 2, 2])
    m_inp = tf.reshape(tf.tile(inputs, [1, 1, M]), [N, L, M, 2])
    locs = tf.reshape(tf.concat([sigxs, sigys], axis=-1), [N, L, M, 2])
    mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=locs, covariance_matrix=covs)

    return tf.reshape(
        tf.log(tf.matmul(
            tf.reshape(mvn.prob(m_inp), [N, L, 1, M]), tf.reshape(weights, [N, L, M, 1])
        )), [N, L, 1]
    )


def weight_sum(inputs, weight_):
    """
    sum the inputs by weights
    :param inputs: N, L, M
    :param weight_: N, L, M
    :return: N, L, 1
    """
    N, L, M = inputs.shape
    return tf.reshape(
            tf.matmul(
                tf.reshape(inputs, [N, L, 1, M]), tf.reshape(weight_, [N, L, M, 1])
            ), [N, L, 1]
        )


def sample_points(logits):
    """
    Transform output parameters to points
    :param logits: [muxs, muys, sigxs, sigys, cors, weights, qs_]
    :return:
    """
    muxs, muys, _, _, _, weights_, qs_ = logits

    x, y = weight_sum(muxs, weights_), weight_sum(muys, weights_)

    q = tf.one_hot(tf.argmax(qs_, axis=2), 3, axis=-1, dtype=tf.float64)
    return tf.concat([x, y, q], axis=2)


def test_function(model, train_x, train_y, train_len, test_output=False, test_nan=False, prev_t=None):
    if test_nan:
        timei = ti()
        if prev_t and np.isnan(prev_t[0]):
            np.set_printoptions(threshold=np.nan)
            print(model.sess.run([model.loss, model.lz, model.sigxs, model.sigys, model.cor],
                                 feed_dict={model.xs: train_x, model.ys: train_y, model.seq_len: train_len}))
            quit()
        else:
            prev_t = model.sess.run([model.loss, model.lz, model.sigxs, model.sigys, model.cor],
                                    feed_dict={model.xs: train_x, model.ys: train_y, model.seq_len: train_len})
            print(prev_t)
        print("Test takes time {}".format(ti() - timei))
        return prev_t
    if test_output:
        last = None
        try:
            tes = model.sess.run(model.rnn_out,
                                 feed_dict={model.xs: train_x, model.ys: train_y, model.seq_len: train_len})
            print(tes.shape)
            last = train_x
        except:
            print(train_x.shape, train_y.shape, train_len.shape)
            tes = model.sess.run(model.rnn_out_all,
                                 feed_dict={model.xs: last, model.ys: train_y, model.seq_len: train_len})
            print(tes.shape)
            quit()
