# Introduction

The data processing parts are in preprocess.py, which can be run or imported.

The training program for each model are in files whose names start with train_.

The train_cnn.py and train_mix_model.py can be run directly, while the train_rnn.py has
different training functions for different model. Choose the corresponding function for
certain model.

All parameters for models can be modified by changing parameters in train function.

## name of the main files:
train_cnn.py, train_rnn.py, train_mix_model.py

# Function of each file
## utils_models
The files in this directory are tensorflow models with repect to its
name. The models are used in training functions.

## training files
The files use the model built in utils_modes and train them by processing
data and certain number of epoches.

## other files
The preprocess.py contains functions for data preprocessing and transformation.

The test.py and sample.py are used to test models or prediction.

# Datasets
All data files are saved in directory named data

## training, eval, *.classes
These are the data files in tfrecord for CNN-RNN models.

## png
These are the 28 bit map data saved in .npy for CNN model downloaded from
https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap.

## simplified
These are the data for all RNN models in json format downloaded from
https://console.cloud.google.com/storage/quickdraw_dataset/full/simplified.

## transformed
These are the data transformed from simplified data into sequences to
feed into RNN models directly.
