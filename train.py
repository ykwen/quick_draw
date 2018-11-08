from utils_models.rnn import *
import numpy as np
from preprocess import *
import os


def mask_seq_len(X, max_l):
    """
    Form input features to certain length and return the true length
    :param X:
    :param max_l:
    :return:
    """
    seq_l = []
    for ind, xx in enumerate(X):
        one_seq_l = len(xx)
        holder = np.array([0, 0, 0, 0, 1]).reshape([1, 5])
        if one_seq_l < max_l:
            X[ind] = np.concatenate([xx, np.repeat(holder, max_l - one_seq_l, axis=0)]).reshape([max_l, 5])
        else:
            X[ind] = np.concatenate([xx[:max_l - 1], holder]).reshape([max_l, 5])
            one_seq_l = max_l
        seq_l.append(one_seq_l)
    return np.stack(X).reshape([X.shape[0], max_l, 5]), np.array(seq_l)


def get_batch(X, Y, batch_size=64, validation_rate=0.2, max_seq_len=100):
    """
    Batch generator with random and shuffle in the begining
    :param X: features
    :param Y: labels
    :param batch_size: as named
    :param validation_rate: as named
    :param max_seq_len: the mean length of points is 45.5, so we set max length to 100
    :return: generator for batch with sequence length
    """
    assert len(X) == len(Y)
    N = len(Y)
    val_size = int(validation_rate * N)
    train_size = N - val_size
    assert val_size >= batch_size
    indexes = np.random.choice(N, N, replace=False)
    X, Y = X[indexes], Y[indexes]
    X, seq_len = mask_seq_len(X, max_seq_len)
    train_x, train_y, val_x, val_y = X[val_size:], Y[val_size:], X[:val_size], Y[:val_size]
    train_seq_len, val_seq_len = seq_len[val_size:], seq_len[:val_size]
    while True:
        train_indexes = np.random.choice(train_size, batch_size, replace=False)
        val_indexes = np.random.choice(val_size, batch_size, replace=False)
        yield train_x[train_indexes], train_y[train_indexes], val_x[val_indexes], val_y[val_indexes],\
              train_seq_len[train_indexes], val_seq_len[val_indexes]


def train_model(X, Y):
    """
    as named
    :param X: features with maxed seq_length
    :param Y: labels
    :return: trained model
    """
    # define parameters here
    batch_size = 256
    num_iteration = 100000
    verbose = 500
    max_len = 100
    params = tf.contrib.training.HParams(
        batch_size=batch_size,
        max_len=max_len,
        one_input_shape=5,
        lr=0.001,
        opt_name="Adam",
        classifier=True,
        bidir=True,
        model="./model/rnn_classifier/diff/bidir",
        best_model="./model/rnn_classifier/diff/best_bidir",
        summary="./model/rnn_classifier/log/diff/bidir",
        rnn_node="lstm",
        num_r_n=512,
        num_r_l=3,
        dr_rnn=0.2,
        num_classes=8,
        restore=False,
        trained_steps=0
    )
    with tf.device("/GPU:0"):
        test_summary = tf.Summary()
        test_summary.value.add(tag='Valid Loss', simple_value=None)
        test_summary.value.add(tag='Valid Accuracy', simple_value=None)

        model = rnn_encoder(params, estimator=False)
        # get batches
        batch_generator = get_batch(X, Y, batch_size=batch_size, max_seq_len=max_len)
        prev_loss = float("inf")
        for i in range(num_iteration):
            train_x, train_y, _, _, train_len, _ = next(batch_generator)
            train_sum, _ = model.sess.run([model.sum, model.step], feed_dict={model.xs: train_x, model.ys: train_y, model.seq_len: train_len})
            model.summary_writer.add_summary(train_sum, i + params.trained_steps)
            model.saver.save(model.sess, params.model)
            if i % verbose == 0 or i == num_iteration - 1:
                valid_loss, valid_acc = model.sess.run([model.loss, model.acc],
                                                       feed_dict={model.xs: train_x, model.ys: train_y, model.seq_len: train_len})
                test_summary.value[0].simple_value = valid_loss
                test_summary.value[1].simple_value = valid_acc
                model.summary_writer.add_summary(test_summary, i + params.trained_steps)
                if prev_loss > valid_loss:
                    model.saver.save(model.sess, params.best_model)
                    prev_loss = valid_loss


if __name__ == '__main__':
    file_path = "./data/simplified"
    save_path = "./data/transformed"
    # choose 8 categories that are totally different by my interest
    diff_categories = ["cat", "angel", "bench", "dragon", "eyeglasses", "ice cream", "t-shirt", "steak"]
    # choose 8 animals as similar categories in sketch
    sim_categories = ["bear", "bird", "cat", "duck", "giraffe", "monkey", "panda", "penguin"]

    # transform these categories
    transformed_file = set(os.listdir(save_path))
    need_to_transform = [f for f in diff_categories + sim_categories if f + ".npy" not in transformed_file]
    if need_to_transform:
        transformed = transform_to_sketch(file_path, need_to_transform)
        save_transformed(transformed, save_path)
    # load target categories
    categories = diff_categories
    X, Y = [], []
    y_dict, y_dict_reverse = {v: k for k, v in enumerate(categories)}, {k: v for k, v in enumerate(categories)}
    for c in diff_categories:
        trans = load_one_transformed(save_path + "/" + c + ".npy")
        X = np.concatenate([X, np.array(trans)])
        ind_test = np.random.choice(len(trans), 1)[0]
        # visualize_one_transformed(trans[ind_test], c)
        Y = np.concatenate([Y, [y_dict[c]] * len(trans)])

    train_model(X, Y)
