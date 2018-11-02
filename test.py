'''Reference: https://www.tensorflow.org/tutorials/sequences/recurrent_quickdraw'''
import sys
import os
import functools
import tensorflow as tf
from utils_models.mixed_classifier import cnn_rnn_classifier as model

# train naive model and test with the data given in tutorial
# fit model to model_fn for training
train_class_file = "./data/training.tfrecord.classes"
train_data_file = "./data/training/training.tfrecord"
eval_data_file = "./data/eval/eval.tfrecord"


def get_num_classes():
    with tf.gfile.GFile(train_class_file, "r") as f:
        classes = [x for x in f]
        num_classes = len(classes)
    return classes, num_classes


# return data processing function for record data
def get_data_fn(mode, record_path, batch_size):
    # parse example record to features
    def _parse_fn(proto, mode):
        # feature mapping dictionary
        feature_dict = {
            "ink": tf.VarLenFeature(dtype=tf.float32),
            "shape": tf.FixedLenFeature([2], dtype=tf.int64)
        }
        # add labels only for training and eval
        if mode != tf.estimator.ModeKeys.PREDICT:
            feature_dict["class_index"] = tf.FixedLenFeature([1], dtype=tf.int64)

        # parse the feature using dictionarys
        features = tf.parse_single_example(proto, feature_dict)
        labels = None
        if mode != tf.estimator.ModeKeys.PREDICT:
            labels = features["class_index"]
            features["ink"] = tf.sparse_tensor_to_dense(features["ink"])
        return features, labels

    def _input_fn():
        # used for tf estimator to transform record to features
        dataset = tf.data.TFRecordDataset.list_files(record_path)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=10) # shuffle datasets
        dataset = dataset.repeat()
        # Preprocesses 10 files as provided concurrently and interleaves records from each file.
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            block_length=1)
        dataset = dataset.map(
            functools.partial(_parse_fn, mode=mode), # fix the model for parse function
            num_parallel_calls=10)
        dataset = dataset.prefetch(10000) # as each class has 10000 data
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=1000) # shuffle datas
        # the original are variable length, so pad them.
        dataset = dataset.padded_batch(
            batch_size, padded_shapes=dataset.output_shapes)
        features, labels = dataset.make_one_shot_iterator().get_next()
        return features, labels
    # return the data processing function
    return _input_fn


# transfer models to model fn for training
def model_fn(features, labels, mode, params):
    # from feature to inputs
    def _get_input(features, labels):
        shape = features["shape"]
        # get real length for each input data
        lengths = tf.squeeze(
            tf.slice(shape, begin=[0, 0], size=[params.batch_size, 1]))
        inputs = tf.reshape(features["ink"], [params.batch_size, -1, 3])
        if labels is not None:
            labels = tf.squeeze(labels)
        return inputs, lengths, labels

    # get data
    x, l, y = _get_input(features, labels)
    # define model
    dnn = model(estimator=True, params=params, inputs=[x, y, l, mode])
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"logits": dnn.logits, "predictions": dnn.prediction},
        loss=dnn.loss,
        train_op=dnn.step,
        eval_metric_ops={"accuracy": dnn.acc})


# create specs for tf train_and_evaluate
def get_specs(params, config, train_data, eval_data, steps):
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=config,
        params=params)

    train_ = tf.estimator.TrainSpec(input_fn=get_data_fn(
        mode=tf.estimator.ModeKeys.TRAIN,
        record_path=train_data,
        batch_size=params.batch_size), max_steps=steps)

    eval_ = tf.estimator.EvalSpec(input_fn=get_data_fn(
        mode=tf.estimator.ModeKeys.EVAL,
        record_path=eval_data,
        batch_size=params.batch_size))

    return estimator, train_, eval_


def run_mixed_model(num_classes):
    params = tf.contrib.training.HParams(
        batch_size=80, num_class=num_classes,
        lr=0.0001, optimizer='Adam',
        num_r_l=3, num_r_n=128, rnn_node='lstm', dr_rnn=0.2,
        num_c_l=(48, 64, 96), ker_cnn=(5, 5, 3), str_cnn=(1, 1, 1),
        bn_cnn=True, dr_cnn=0.3
    )

    model_dir = "model/mixed_model/test"
    train_dir = "data/training/training.tfrecord-?????-of-?????"
    eval_dir = "data/eval/eval.tfrecord-?????-of-?????"

    config = tf.estimator.RunConfig(model_dir=model_dir,
                                    save_checkpoints_secs=300,
                                    save_summary_steps=100)

    max_steps = 1000000
    est, tr_, ev_ = get_specs(params, config, train_dir, eval_dir, max_steps)
    tf.estimator.train_and_evaluate(est, tr_, ev_)


if __name__ == '__main__':
    classes, num_classes = get_num_classes()
    with tf.device('/GPU:0'):
        run_mixed_model(num_classes)
