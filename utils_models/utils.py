# functions for calculation
import tensorflow as tf
import math


def bivariate_normal(inputs, params):
    '''
    :param inputs: (N, seq_length, 2)
    :param params: muxs, muys, sigxs, sigys, cors, weights, qs, (M,)
    :return: (N, L)
    '''
    muxs, muys, sigxs, sigys, cors, weights, _ = params
    input = tf.reshape(inputs, [-1, 2]) # N_max, 2
    n_max = input.shape[0]
    M = muxs.shape[0]

    x, y = tf.slice(input, [0, 0], [n_max, 1]), tf.slice(input, [0, 1], [n_max, 1])
    x, y = tf.tile(x, [1, M]), tf.tile(y, [1, M]) # N_max, M

    x_mux, y_muy = tf.subtract(x, muxs), tf.subtract(y, muys)
    sigxs_2, sigys_2, sigx_y = tf.square(sigxs), tf.square(sigys), tf.multiply(sigxs, sigys)
    _cor_sq = 1 - tf.square(cors)

    z = tf.square(x_mux) / sigxs_2 - \
        tf.multiply(tf.multiply(tf.multiply(2, cors), x_mux), y_muy) + \
        tf.square(y_muy) / sigys_2
    tmp = tf.multiply(tf.multiply(tf.multiply(2 * math.pi, sigxs), sigys), tf.sqrt(_cor_sq))
    p = tf.multiply(tf.divide(1, tmp), tf.exp(-tf.divide(z, tf.multiply(2, _cor_sq)))) # N_max, M
    return tf.matmul(p, tf.reshape(weights, [M, 1]))


def sample_fn(params, de_out):
    '''
    first implementing in tf helper.py, then modified to use for transform output
    :param de_out: decoder raw outputs
    :return: [N, L, (x, y, p1, p2, p3)]
    '''
    n, m = params.batch_size, params.gmm_dim
    muxs = tf.slice(de_out, [0, 0], [n, m])
    muys = tf.slice(de_out, [0, m], [n, m])
    weights_ = tf.transpose(tf.slice(de_out, [0, 5 * m], [n, m]))
    qs_ = tf.slice(de_out, [0, 6 * m], [n, 3])

    x, y = tf.trace(tf.matmul(muxs, weights_)), tf.trace(tf.matmul(muys, weights_))
    x, y = tf.reshape(x, [params.batch_size, 1]), tf.reshape(y, [params.batch_size, 1])
    q = tf.one_hot(tf.argmax(qs_, axis=1), 3)
    return tf.concat([x, y, q], axis=1)
