import tensorflow as tf
import numpy as np
from utils_models.rnn import *
from preprocess import visualize_one_transformed


def sample_decoder(model_path, data_save_path, category, num_sample):
    """
    Using the trained model to sample image of given category
    :param model_path: general model saved path
    :param data_save_path: where to save sampled model
    :param category: the model category
    :param num_sample: the number of image to sample
    :return: sampled images' points
    """
    max_len = 100
    params = tf.contrib.training.HParams(
        batch_size=num_sample,
        max_len=max_len,
        one_input_shape=5,
        model=model_path.format(category, category),
        lr=0.0001,
        opt_name="Adam",
        classifier=False,
        bidir=False,
        rnn_node="lstm",
        num_r_n=2048,
        gmm_dim=128,
        num_r_l=1,
        activation=tf.nn.tanh,
        dr_rnn=0.1,
        num_classes=8,
        clip_gradients=1.,
        mode=tf.estimator.ModeKeys.PREDICT,
        temper=1.,
        w_KL=1.,
        eta_min=0.01,
        R=0.99999,
        kl_min=0.20,
        train_with_inputs=False,
        restore=True,
        trained_steps=0
    )
    with tf.device("/GPU:0"):
        tf.reset_default_graph()
        model = RNNDecoder(params, decoder=True)
        points = model.sess.run(model.logits)
        for p in points:
            visualize_one_transformed(p)
        np.save(data_save_path.format(category, num_sample), points)


if __name__ == '__main__':
    model_path = "./model/rnn_decoder/{}/{}"
    data_save_path = "./data/decoder_sampled/{}_{}"

    sample_decoder(model_path, data_save_path, 'cat', 5)
