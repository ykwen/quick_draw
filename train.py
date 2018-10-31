import tensorflow as tf
from utils_models import mixed_classifier, cnn, rnn, sketch_rnn

with tf.device('/gpu:0'):
    model1 = mixed_classifier.cnn_rnn_classifier((10, 9, 8), 2, True)
