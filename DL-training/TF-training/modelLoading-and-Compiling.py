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

## result is a .pb compiled frozen graph compatible with Roger's app. 
