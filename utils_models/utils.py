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

    sigxs_1, sigys_1 = tf.reshape(sigxs, [-1, 1]), tf.reshape(sigys, [-1, 1])
    sigxs_2, sigys_2, cors_ = tf.square(sigxs_1), tf.square(sigys_1), \
                              tf.reshape(tf.multiply(tf.multiply(sigxs, sigys), cors), [-1, 1])
    covs = tf.reshape(tf.concat([sigxs_2, cors_, cors_, sigys_2], axis=1), [-1, 2, 2])
    m_inp = tf.reshape(tf.tile(inputs, [1, 1, M]), [-1, 2])
    locs = tf.concat([sigxs_1, sigys_1], axis=1)
    mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=locs, covariance_matrix=covs)

    return tf.reshape(
        tf.log(tf.matmul(
            tf.reshape(mvn.prob(m_inp), [N, L, 1, M]), tf.reshape(weights, [N, L, M, 1])
        )), [N, L, 1]
    )


def bivariate_normal_own(inputs, params):
    """
    Using given function in tensorflow is convienent, but it takes 250s for 1 step, that's too slow
    While, the gpu usage is almost 0
    :param inputs: (N, max_seq_length, 2)
    :param params: muxs, muys, sigxs, sigys, cors, weights, qs, (N, L, M)
    :return: log (N, L, 1)
    """
    with tf.device("/GPU:0"):
        muxs, muys, sigxs, sigys, cors, weights, _ = params
        N, L = inputs.shape[0], inputs.shape[1]
        M = muxs.shape[-1]

        x, y = tf.slice(inputs, [0, 0, 0], [N, L, 1]), tf.slice(inputs, [0, 0, 1], [N, L, 1])
        x, y = tf.tile(x, [1, 1, M]), tf.tile(y, [1, 1, M])

        x, y, = tf.reshape(x, [-1, 1]), tf.reshape(y, [-1, 1])
        sigxs_1, sigys_1 = tf.reshape(sigxs, [-1, 1]), tf.reshape(sigys, [-1, 1])
        sigxs_2, sigys_2, cors_ = tf.square(sigxs_1), tf.square(sigys_1), \
                                  tf.reshape(tf.multiply(tf.multiply(sigxs, sigys), cors), [-1, 1])
        covs = tf.reshape(tf.concat([sigxs_2, cors_, cors_, sigys_2], axis=1), [-1, 2, 2])

        # Calcualte for N*L*M values which is wrong again
        inp = tf.reshape(tf.concat([x - sigxs_1, y - sigys_1], axis=1), [-1, 1, 2])
        log_Z = tf.reshape(tf.log(2 * math.pi) + 0.5 * tf.linalg.logdet(covs), [-1, 1])
        tmp_y = tf.reshape(tf.matmul(tf.matmul(inp, tf.linalg.inv(covs)), tf.transpose(inp, [0, 2, 1])), [-1, 1])

        log_p = tf.reshape(-0.5 * tmp_y - log_Z, [N, L, 1, M])

        return tf.reshape(
            tf.matmul(
                tf.reshape(log_p, [N, L, 1, M]), tf.reshape(weights, [N, L, M, 1])
            ), [N, L, 1]
        )


def weight_sum(inputs, weight_):
    """
    sum the inputs by weights
    :param inputs: N, L, M
    :param weight_: N, L, M
    :return: N, L, 1
    """
    return tf.reshape(
        tf.linalg.diag_part(
            tf.matmul(inputs, tf.transpose(weight_, [0, 2, 1]))
        ), [inputs.shape[0], inputs.shape[1], 1]
    )


def sample_points(logits):
    """
    Transform output parameters to points
    :param logits: [muxs, muys, sigxs, sigys, cors, weights, qs_]
    :return:
    """
    muxs, muys, _, _, _, weights_, qs_ = logits

    x, y = weight_sum(muxs, weights_), weight_sum(muys, weights_)

    q = tf.one_hot(tf.argmax(qs_, axis=2), 3, axis=-1)
    return tf.concat([x, y, q], axis=2)


def test_function(model, train_x, train_y, train_len, test_output=True, test_nan=False):
    if test_nan:
        timei = ti()
        tes = model.sess.run(model.loss,
                             feed_dict={model.xs: train_x, model.ys: train_y, model.seq_len: train_len})
        if np.isnan(tes):
            print(model.sess.run([model.logn, model.Ls, model.Ls, model.Lp, model.L_kl],
                                 feed_dict={model.xs: train_x, model.ys: train_y, model.seq_len: train_len}))
            quit()
        else:
            prev_t = model.sess.run([model.Ls, model.Lp, model.L_kl],
                                    feed_dict={model.xs: train_x, model.ys: train_y, model.seq_len: train_len})
            print(prev_t)
        print("Test takes time {}".format(ti() - timei))
    if test_output:
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
