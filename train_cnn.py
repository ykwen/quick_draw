import numpy as np
import tensorflow as tf
from preprocess import transform_to_png
import matplotlib.pyplot as plt
from utils_models.cnn import *


def load_png_py(path, categories, cate_dict):
    data = []
    for c in categories:
        pngs = np.load(path + "/" + c + ".npy")
        n = len(pngs)
        pngs = pngs.reshape([n, 28, 28, 1])
        label = np.repeat(cate_dict[c], len(pngs))
        data.append((pngs, label))
    return data


def get_batch(data, size, valid_rate):
    X, Y = [], []
    for d in data:
        X.append(d[0]), Y.append(d[1])
    assert len(X) == len(Y)
    X, Y = np.concatenate(X), np.concatenate(Y)
    N, num_reshuffle, count, valid_size = len(X), int(len(X) / size), int(len(X) / size), int(len(X) * valid_rate)
    assert valid_size >= size

    indexes = np.random.choice(N, N, replace=False)
    X, Y = X[indexes], Y[indexes]
    train_x, train_y, val_x, val_y = X[valid_size:], Y[valid_size:], X[:valid_size], Y[:valid_size]
    while True:
        if count == num_reshuffle:
            train_idx = np.random.choice(N - valid_size, N - valid_size, replace=False)
            train_x, train_y = train_x[train_idx], train_y[train_idx]
            count = 0
        val_idx = np.random.choice(valid_size, size, replace=False)
        yield train_x[count * size: count * size + size], train_y[count * size: count * size + size],\
              val_x[val_idx], val_y[val_idx]


def get_cnn(name, batch_size, num_class, params):
    if name == "res18":
        return ResNet_18(batch_size, num_class, params)
    else:
        raise ValueError("Model is not implemented yet")


def train_cnn(data, cnn_type, model_save_path):
    batch_size = 64
    valid_rate = 0.1
    num_iteration = 10000
    save_every = 50
    verbose = 200
    num_class = len(data)
    params = tf.contrib.training.HParams(
                batch_size=batch_size,
                lr=0.001,
                decay_step=100,
                decay_rate=0.25,
                clip_gradients=1.0,
                opt_name="Adam",
                classifier=True,
                model=model_save_path,
                best_model=model_save_path,
                summary=model_save_path,
                dr_rnn=0.1,
                mode=tf.estimator.ModeKeys.TRAIN,
                num_classes=8,
                restore=False,
                trained_steps=0
    )

    batch_generator = get_batch(data, batch_size, valid_rate)
    model = get_cnn(cnn_type, batch_size, num_class, params)

    test_summary = tf.Summary()
    test_summary.value.add(tag='Valid Loss', simple_value=None)
    if params.classifier:
        test_summary.value.add(tag='Valid Accuracy', simple_value=None)
    prev_loss = float("inf")
    for i in range(num_iteration):
        train_x, train_y, val_x, val_y = next(batch_generator)

        train_sum, _ = model.sess.run([model.sum, model.step],
                                      feed_dict={model.xs: train_x, model.ys: train_y})
        model.summary_writer.add_summary(train_sum, i + params.trained_steps)

        if i % save_every == 0 or i == num_iteration - 1:
            model.saver.save(model.sess, params.model)
        if i % verbose == 0 or i == num_iteration - 1:
            valid_loss = model.sess.run(model.loss,
                                        feed_dict={model.xs: val_x, model.ys: val_y})

            test_summary.value[0].simple_value = valid_loss
            if params.classifier:
                valid_acc = model.sess.run(model.acc,
                                           feed_dict={model.xs: val_x, model.ys: val_y})
                test_summary.value[1].simple_value = valid_acc
            print("Now at iteration {}, valid loss is {}, valid accuracy is {}".format(i, valid_loss, valid_acc))
            model.summary_writer.add_summary(test_summary, i + params.trained_steps)
            if prev_loss > valid_loss:
                model.saver.save(model.sess, params.best_model)
                prev_loss = valid_loss

    return model


if __name__ == '__main__':
    file_path = "./data/png"
    diff_categories = ["cat", "angel", "bench", "dragon", "eyeglasses", "ice cream", "t-shirt", "steak"]
    sim_categories = ["bear", "bird", "cat", "duck", "giraffe", "monkey", "panda", "penguin"]
    cate_type = "diff"

    if cate_type == "diff":
        categories = diff_categories
    elif cate_type == "sim":
        categories = sim_categories
    else:
        raise ValueError("Invalid Data Type")

    cate_dict = {c: i for i, c in enumerate(categories)}
    cate_dict_rev = {i: c for i, c in enumerate(categories)}
    png_data = load_png_py(file_path, categories, cate_dict)

    cnn_type = "res18"
    model_save_path = "./model/cnn_classifier_tmp/{}/{}".format(cate_type, cnn_type)
    train_cnn(png_data, cnn_type, model_save_path)
