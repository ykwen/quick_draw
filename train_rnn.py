from utils_models.rnn import *
from utils_models.utils import test_function
from preprocess import *
import os

np.random.seed(233)


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


def generate_Y(X):
    """
    Shift X to generate Y
    :param X: input (N, 100, 5)
    :return: shifted input (N, 100, 5)
    """
    holder = np.array([0, 0, 0, 0, 1]).reshape([1, 5])
    return np.concatenate([X[:, 1:, :], np.repeat(holder, len(X), axis=0).reshape([len(X), 1, 5])], axis=1)


def normalize(X):
    """
    points normalize
    :param X: input (N, 100, 5)
    :return: normalized input (N, 100, 5)
    """
    input = X[:, :, :2].reshape([-1, 1])
    mean = np.mean(input)
    dev = np.std(input)
    std_X = (X[:, :, :2] - mean) / dev
    return np.concatenate([std_X, X[:, :, 2:]], axis=2)


def data_augment(X, Y, seq_len):
    """
    Augment data by multiplying 1.1 and 0.9 randomly
    :param X: delta X
    :param Y: delta Y
    :param seq_len: seq_length
    :return: augmented X, Y
    """
    assert len(X) == len(Y)
    N, L = len(X), len(X[0])
    mask_X = np.random.choice(a=[0.9, 1.1], size=[N, L, 2], p=[0.5, 0.5])
    mask_Y = np.random.choice(a=[0.9, 1.1], size=[N, L, 2], p=[0.5, 0.5])
    masked_X, masked_Y = np.multiply(mask_X, X[:, :, :2]), np.multiply(mask_Y, Y[:, :, :2])
    masked_X, masked_Y = np.concatenate([masked_X, X[:, :, 2:]], axis=2), np.concatenate([masked_Y, Y[:, :, 2:]], axis=2)
    return np.concatenate([X, masked_X]), np.concatenate([Y, masked_Y]), np.concatenate([seq_len, seq_len])


def get_batch(X, Y=None, batch_size=64, validation_rate=0.1, max_seq_len=100, augment=False):
    """
    Batch generator with random and shuffle in the begining
    :param X: features
    :param Y: labels, None for decoder
    :param batch_size: as named
    :param validation_rate: as named
    :param max_seq_len: the mean length of points is 45.5, so we set max length to 100
    :param augment: Whether to augment data with t
    :return: generator for batch with sequence length, if decoder, Y is shifted input
    """
    X, seq_len = mask_seq_len(X, max_seq_len)
    X = normalize(X)
    if Y is None:
        Y = generate_Y(X)
    if augment:
        X, Y, seq_len = data_augment(X, Y, seq_len)
    assert len(X) == len(Y)
    N = len(Y)
    val_size = int(validation_rate * N)
    train_size = N - val_size
    assert val_size >= batch_size
    indexes = np.random.choice(N, N, replace=False)
    X, Y = X[indexes], Y[indexes]
    assert not (np.any(np.isnan(X)) and np.any(np.isnan(Y)))
    train_x, train_y, val_x, val_y = X[val_size:], Y[val_size:], X[:val_size], Y[:val_size]
    train_seq_len, val_seq_len = seq_len[val_size:], seq_len[:val_size]
    while True:
        train_indexes = np.random.choice(train_size, batch_size, replace=False)
        val_indexes = np.random.choice(val_size, batch_size, replace=False)
        yield train_x[train_indexes], train_y[train_indexes], val_x[val_indexes], val_y[val_indexes],\
              train_seq_len[train_indexes], val_seq_len[val_indexes]


def train_model(model, params, num_iteration, save_every, verbose, X, Y=None, augment=False):
    """
    Train the given model with given params
    :param model: rnn_encoder, rnn_decoder, sktch_rnn, cnn...
    :param params: model's params
    :param num_iteration: num_iteration
    :param save_every: save the model every constant
    :param verbose: Save the best model or not
    :param X: input features
    :param Y: input targets
    :return: trained model
    """
    test_summary = tf.Summary()
    test_summary.value.add(tag='Valid Loss', simple_value=None)
    if params.classifier:
        test_summary.value.add(tag='Valid Accuracy', simple_value=None)

    # get batches
    batch_generator = get_batch(X, Y, batch_size=params.batch_size, max_seq_len=params.max_len, augment=augment)
    prev_loss = float("inf")

    prev = None
    for i in range(num_iteration):
        train_x, train_y, val_x, val_y, train_len, val_len = next(batch_generator)
        # print(train_x.shape, train_y.shape, train_len.shape, val_x.shape)
        # prev = test_function(model, train_x, train_y, train_len, test_nan=True, prev_t=prev)
        # quit()
        # timei = ti()

        train_sum, _ = model.sess.run([model.sum, model.step],
                                      feed_dict={model.xs: train_x, model.ys: train_y, model.seq_len: train_len})
        model.summary_writer.add_summary(train_sum, i + params.trained_steps)

        # print("Trained one batch time {}".format(ti() - timei))

        if i % save_every == 0 or i == num_iteration - 1:
            model.saver.save(model.sess, params.model)
        if i % verbose == 0 or i == num_iteration - 1:
            valid_loss = model.sess.run(model.loss,
                                        feed_dict={model.xs: val_x, model.ys: val_y,
                                                   model.seq_len: val_len})

            test_summary.value[0].simple_value = valid_loss
            if params.classifier:
                valid_acc = model.sess.run(model.acc,
                                           feed_dict={model.xs: val_x, model.ys: val_y,
                                                      model.seq_len: val_len})
                test_summary.value[1].simple_value = valid_acc
            model.summary_writer.add_summary(test_summary, i + params.trained_steps)
            if prev_loss > valid_loss:
                model.saver.save(model.sess, params.best_model)
                prev_loss = valid_loss

    return model


def train_encoder_model(file_path, save_path):
    """
    Define model parameters and train model
    :param file_path: data file path
    :param save_path: transformed data file path
    :return: trained model
    """
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
    cates = diff_categories
    cate_type = "diff"
    X, Y = [], []
    y_dict, y_dict_reverse = {v: k for k, v in enumerate(cates)}, {k: v for k, v in enumerate(cates)}
    for c in cates:
        trans = load_one_transformed(save_path + "/" + c + ".npy")
        X = np.concatenate([X, np.array(trans)])
        # visualize_one_transformed(trans[ind_test], c)
        Y = np.concatenate([Y, [y_dict[c]] * len(trans)])

    # define parameters here
    batch_size = 256
    num_iteration = 20000
    save_every = 50
    verbose = 100
    max_len = 100
    for model_type in ["bidir", "basic"]:
        for l in range(1, 4):
            model_layer = model_type + "_{}".format(l)
            params = tf.contrib.training.HParams(
                batch_size=batch_size,
                max_len=max_len,
                one_input_shape=5,
                lr=0.0001,
                decay_step=100,
                decay_rate=0.25,
                clip_gradients=1.0,
                opt_name="Adam",
                classifier=True,
                bidir=model_type == "bidir",
                model="./model/rnn_classifier/{}/{}/{}".format(cate_type, model_type, model_layer),
                best_model="./model/rnn_classifier/{}/{}/best_{}".format(cate_type, model_type, model_layer),
                summary="./model/rnn_classifier/log/{}/{}".format(cate_type, model_layer),
                rnn_node="lstm",
                num_r_n=512,
                num_r_l=l,
                activation=tf.nn.tanh,
                dr_rnn=0.1,
                num_classes=8,
                restore=False,
                trained_steps=0
            )
            with tf.device("/GPU:0"):
                tf.reset_default_graph()
                model = RNNEncoder(params, estimator=False)
                train_model(model, params, num_iteration, save_every, verbose, X, Y)


def train_decoder_model(file_path, save_path, category):
    """
    Train one category decoder model at a time
    :param file_path: data file path
    :param save_path: transformed data file path
    :param category: the category
    :return: trained model
    """
    transformed_file = set(os.listdir(save_path))
    if category + ".npy" not in transformed_file:
        data = transform_to_sketch(file_path, [category])
        save_transformed(data, save_path)
    else:
        data = load_one_transformed(save_path + "/" + category + ".npy")

    batch_size = 128
    num_iteration = 30000
    save_every = 20
    verbose = 40
    max_len = 100
    params = tf.contrib.training.HParams(
        batch_size=batch_size,
        max_len=max_len,
        one_input_shape=5,
        lr=0.0001,
        opt_name="Adam",
        classifier=False,
        bidir=False,
        model="./model/rnn_decoder/{}/{}".format(category, category),
        best_model="./model/rnn_decoder/{}/{}_best".format(category, category),
        summary="./model/rnn_decoder/log/{}".format(category),
        rnn_node="lstm",
        num_r_n=2048,
        gmm_dim=20,
        num_r_l=1,
        activation=tf.nn.tanh,
        dr_rnn=0.1,
        num_classes=8,
        clip_gradients=1.,
        mode=tf.estimator.ModeKeys.TRAIN,
        temper=1.,
        w_KL=1.,
        eta_min=0.01,
        R=0.99999,
        kl_min=0.20,
        train_with_inputs=True,
        restore=False,
        trained_steps=0
    )
    with tf.device("/GPU:0"):
        tf.reset_default_graph()
        model = RNNDecoder(params, decoder=True)
        train_model(model, params, num_iteration, save_every, verbose, data, augment=True)


if __name__ == '__main__':
    data_path = "./data/simplified"
    data_save_path = "./data/transformed"

    # train_encoder_model(data_path, save_path)
    train_decoder_model(data_path, data_save_path, "cat")
