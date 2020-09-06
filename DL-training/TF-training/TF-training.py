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

def mobilenetv2_transfer_learning(num_classes):
    ft_layers = [2] 
    mobilenet = keras.applications.MobileNetV2()
    x = mobilenet.layers[-3].output 

    x = Convolution2D(64, (3, 3), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5, name='dropout')(x)
    predictions = Dense(num_classes, activation='softmax', name='re_lu_1/Relu6')(x)

    model = Model(inputs=mobilenet.input, outputs=predictions)

    print(model.summary())
    return model, ft_layers

def fine_tune(model, ft_depth):
    for layer in model.layers[-ft_depth:]:
        layer.trainable = True
    for layer in model.layers[:-ft_depth]:
        layer.trainable = False

    # Show which layers are now trainable
    for layer in model.layers:
        if layer.trainable:
            print(layer)
    return model

### function taking in two numpy nd-arrays (predictions & true labels) and outputing the accuracies (raw accuracy,
### off-by-1 accuracy, off-by-2 accuracy) ###
def getAccuracy(preds, labels):
    rightPredCount = 0 ## counter for right predictions
    offBy1Count = 0 ## counter for off-by-1 predictions
    offBy2Count = 0 ## counter for off-by-2 predictions
    accDico = {
        "correctPredictions": '', 
        "raw accuracy": '',
        "offBy1Predictions":'',
        "offBy1acc": '',
        "offBy2Predictions":'',
        "offBy2acc": ''
    }
    if len(preds) != len(labels):
        raise ValueError("Arrays must have the same size")
        return
    else:
        for i in range(len(preds)): # iterate through arrays
            if preds[i] == labels[i]: # correct prediction
                rightPredCount += 1
            elif abs( preds[i] - labels[i] ) == 1:
                offBy1Count += 1
            elif abs( preds[i] - labels[i] ) == 2:
                offBy2Count += 1
            else:
                bs = 0 # do nothing
    rawAcc = float( rightPredCount / len(preds) )
    offBy1acc = float( (rightPredCount + offBy1Count) / len(preds) )
    offBy2acc = float( (rightPredCount + offBy1Count + offBy2Count) / len(preds) )
    accDico['correctPredictions'] = rightPredCount
    accDico['raw accuracy'] = rawAcc
    accDico['offBy1Predictions'] = offBy1Count
    accDico['offBy1acc'] = offBy1acc
    accDico['offBy2Predictions'] = offBy2Count
    accDico['offBy2acc'] = offBy2acc
    return accDico

def numImagesTotal():
    notebook_path = os.path.dirname(os.path.realpath('__file__'))
    streetViewData_dir = "/GoogleStreetView_images/labelled_data_already_scp-ed"
    path = notebook_path + streetViewData_dir
    print(path)
    dirs = os.listdir(path)
    dirs.sort()
    totalNumImages = 0
    sizesPerLabel = {}
    for k in dirs:
        path2 = path + '/' + k
        files = os.listdir(path2)
        sizesPerLabel[k] = len(files)
        totalNumImages += len(files)
    print(sizesPerLabel)
    print("Total number of images in the directory: ", totalNumImages)
    print("---")
    return path, totalNumImages, sizesPerLabel

train_path = 'GoogleStreetView_images/3datasets/train'
val_path   = 'GoogleStreetView_images/3datasets/val'
test_path  = 'GoogleStreetView_images/new_test_set-never_seen'
notebook_path = os.path.dirname(os.path.realpath('__file__'))
os.path.isdir(test_path)

### create batches ###
preprocess=True
class_mode="categorical"
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
if preprocess:
    preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input
else:
    preprocess_fn=None
    
train_batches = ImageDataGenerator(preprocessing_function=preprocess_fn).flow_from_directory(train_path, 
                                                         target_size=(224,224), 
                                                         classes=['0','1','2','3','4','5','6','7'], batch_size=64,
                                                                                            class_mode=class_mode)   
val_batches = ImageDataGenerator(preprocessing_function=preprocess_fn).flow_from_directory(val_path, 
                                                         target_size=(224,224), 
                                                         classes=['0','1','2','3','4','5','6','7'], batch_size=32,
                                                                                          class_mode=class_mode)
test_batches = ImageDataGenerator(preprocessing_function=preprocess_fn).flow_from_directory(test_path, 
                                                         target_size=(224,224), 
                                                         classes=['0','1','2','3','4','5','6','7'], batch_size=64,
                                                                                           class_mode=class_mode)
print(train_batches.num_classes)
print(train_batches.samples)

steps_per_epoch = train_batches.samples // train_batches.batch_size
validation_steps = val_batches.samples // val_batches.batch_size

## START ##
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
print(IMG_SHAPE)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.summary()

# defining layers
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(8, activation='softmax', name="act_softmax")
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

### MODEL ###
print(IMG_SHAPE)
inputs = base_model.input # base_model is MobileNetV2 pretrained on Imagenet
x = base_model(inputs, training=True) # use training=False as our model contains a BatchNormalization layer
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

base_learning_rate = 0.0001
model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics = ['accuracy'])
model.summary()

## save model (architecture + weights ##
import pydot
import pydotplus
notebook_path = os.path.dirname(os.path.realpath('__file__'))
print(os.path.isdir(notebook_path + '/model/architecture/'))
 
model_visual_fp = notebook_path + '/graph/'+ 'mobilenetv2' + '.png'
print("Saving model visual to file: "+ model_visual_fp)
if not os.path.isdir(notebook_path + '/graph'):
    os.mkdir(notebook_path + '/graph')
tf.keras.utils.plot_model(model, to_file=model_visual_fp, show_shapes=True)

## Saving the architecture / configuration only (explicit graphs of layers)
architecture_fp = notebook_path + "/model/architecture/"+ 'mobilenetv2' + '.json'
print("Saving model architecture to file: "+architecture_fp)
model_json_config = model.to_json()
with open(architecture_fp, "w") as json_file:
    json.dump(model_json_config, json_file)

# checkpoint 
if not os.path.isdir(notebook_path + '/model/weights'):
    os.mkdir(notebook_path + '/model/weights')
checkpoint_filepath = notebook_path + '/model/weights/'
model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath + 'mobilenetv2' + '-{epoch:02d}-{val_acc:.2f}.hdf5',
        monitor='val_acc',
        verbose=1, 
        save_weights_only=True,
        save_best_only=True, 
        mode='max'
    )

total_epochs = 15
start_time = time.time()
print("Program starts at time: ",start_time)

history = model.fit_generator(train_batches,
                         epochs=total_epochs,
                         steps_per_epoch=steps_per_epoch,
                         validation_data=val_batches,
                         validation_steps=validation_steps,
                         callbacks=[model_checkpoint]
                        )  
end_time = time.time()
print("Program ends at time",end_time) 
total_time = (end_time - start_time) 
print("Total time elapsed in training(s): " +str("%.3f" %total_time))  

# save history
print("Saving history to file")
notebook_path = os.path.dirname(os.path.realpath('__file__'))
 
if not os.path.isdir(notebook_path + '/graph'):
    os.mkdir('./graph')
with open('./graph/history_RUN'+ '.pkl','wb') as f:
    pickle.dump(history.history,f)
print(history.history)

notebook_path = os.path.dirname(os.path.realpath('__file__'))
RUN_ID = 3
# refer to train_result.json
architecture_fp = notebook_path +'/model/architecture/mobilenetv2.json'
weights_fp = notebook_path +'/model/weights/mobilenetv2-02-0.57.hdf5'
save_fp = notebook_path + '/model/saved_pb/'+'run' + str(RUN_ID)+".pb"
model_type = 'V2'

### START SESSION & LOAD MODEL ###
sess = tf.Session()
final_layer = 'act_softmax/Softmax'
with open(architecture_fp,'r') as f:
    model_json = json.load(f)
print(final_layer)
model = tf.keras.models.model_from_json(model_json)
model.load_weights(weights_fp)
print("Done")

session = keras.backend.get_session()
init = tf.global_variables_initializer()
session.run(init)
# session = K.get_session()
minimal_graph = graph_util.convert_variables_to_constants(session, session.graph.as_graph_def(), [final_layer])
print("Saving to: " + save_fp)
graph_io.write_graph(minimal_graph, '.', save_fp, as_text=False) 

