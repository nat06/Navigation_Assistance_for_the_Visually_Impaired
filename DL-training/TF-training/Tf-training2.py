from __future__ import absolute_import, division, print_function
from __future__ import print_function
from __future__ import division
import sys  
import numpy as np
import matplotlib.pyplot as plt
import time
import os, shutil
from shutil import copyfile
import copy
import PIL
from PIL import Image
import IPython.display as display
import csv
import random
import pathlib
import cProfile
import time
import tensorflow as tf
from keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from matplotlib import pyplot as plt
import time
import os
import pickle
import json
from datetime import date
import numpy
from keras.utils import plot_model
import pickle
import json 
import keras
from keras import Model
from keras.layers import Convolution2D, Activation, GlobalAveragePooling2D, Reshape, Dropout, Dense, ReLU
import numpy as np
import tensorflow as tf
import keras.backend as K
from tensorflow.python.framework import graph_io, graph_util
from keras import backend as K
from keras.models import load_model
from keras.models import model_from_json

device = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
print(device)

### function returns the number of files in the subdirectory structure ###
def howManyFiles(path):
    notebook_path = os.path.dirname(os.path.realpath('__file__'))
    rootDir = notebook_path + path
    subDirs = os.listdir(rootDir)
    count = 0
    for sub in subDirs:
        files = os.listdir(rootDir + sub)
        count += len(files)
    print(path, " contains ", count, " images.")
    return count

import tensorflow_hub as hub
print("Hub version:", hub.__version__)

module_selection = ("mobilenet_v2_100_224", 224) 
handle_base, pixels = module_selection
MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

### different way of loading the images ###
import itertools
BATCH_SIZE = 120
# BATCH_SIZE = 64
num_classes = 8
IMG_HEIGHT = 224 # the image height to be resized to
IMG_WIDTH = 224  # the image width to be resized to
CHANNELS = 3 # The 3 color channels
notebook_path = os.path.dirname(os.path.realpath('__file__')) 
streetViewData_dirABS = "/GoogleStreetView_images/labelled_data"
streetViewData_dirFULL = notebook_path + streetViewData_dirABS 
streetViewData_dir = "./GoogleStreetView_images/labelled_data"
image_count = howManyFiles(streetViewData_dirABS)
print("---------------")

data_dir = pathlib.Path(streetViewData_dirFULL)
image_count = len(list(data_dir.glob('*/*.jpg')))
print("image_count: ", image_count)
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
CLASS_NAMES = np.sort(CLASS_NAMES)
print(CLASS_NAMES)
print("-------STARTING--------")
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH) 
datagen_kwargs = dict(rescale=1./255, validation_split=0.1)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, interpolation="bilinear")

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs) # does train/val split from a single directory
print("valid datagen: ", valid_datagen)
valid_generator = valid_datagen.flow_from_directory(data_dir, subset="validation", shuffle=True, **dataflow_kwargs)
print("Type of valid_generator: ", type(valid_generator))
for image_batch, label_batch in valid_generator:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break
do_data_augmentation = False 
if do_data_augmentation:
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        horizontal_flip=True,
        width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2,
        **datagen_kwargs)
else: 
    print("here")
    train_datagen = valid_datagen
train_generator = train_datagen.flow_from_directory(data_dir, subset="training", shuffle=True, **dataflow_kwargs)
print(int(0.1*image_count)) 
print("train_generator.samples: ", train_generator.samples)
print("valid_generator.samples: ", valid_generator.samples)


do_fine_tuning = True  ## enable for greater accuracy
print(IMAGE_SIZE + (3,))
model = tf.keras.Sequential([
    # Explicitly define the input shape so the model can be properly
    # loaded by the TFLiteConverter
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer("./modelWGET", trainable=do_fine_tuning), 
    # can do Trainable=False, modifies the # of trainable parameters
    # trainable=True for fine-tuning, trainable=False for feature-extraction
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(60, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(train_generator.num_classes)
])
model.build((None,)+IMAGE_SIZE+(3,)) # batch input shape
model.summary()

from keras.callbacks import ModelCheckpoint

NUM_EPOCHS = 30
notebook_path = os.path.dirname(os.path.realpath('__file__'))
checkpoint_filepath = notebook_path + '/TF_retrained_models/latest/' 
print(checkpoint_filepath)

model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath + 'model.{epoch:02d}',
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1,
    save_freq='epoch')

model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.06, momentum=0.7),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
    metrics=['accuracy']
) 
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = valid_generator.samples // valid_generator.batch_size

hist = model.fit(
train_generator,
epochs=NUM_EPOCHS, steps_per_epoch=steps_per_epoch,
validation_data=valid_generator,
validation_steps=validation_steps,
callbacks=[model_checkpoint_callback]).history

checkpoint_filepath = notebook_path + '/TF_retrained_models/latest/'
new_model = tf.keras.models.load_model(checkpoint_filepath + '...')
new_model.summary
