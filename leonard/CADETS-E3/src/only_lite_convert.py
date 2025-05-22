# 将 keras 模型转换为 tf lite 格式，更小且运行更快
keras_file="./lite.h5"
import keras
import json
import keras.backend as K
import tensorflow as tf
import numpy as np
import models
import operator
from collections import defaultdict
import pickle
import sys
import argparse
from time import *
parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-model', action='store', dest='model',
                    help='param file file')
parser.add_argument('-param', action='store',dest='param')
parser.add_argument('-sequence_length', default='10', action='store',dest='sequence_length')
args = parser.parse_args()
# args = parser.parse_args([
#     '-model', './testLeonard/src/vertex.hdf5',
#     '-param', './testLeonard/data/vertex.params.json',
#     '-sequence_length', '10'
# ])
#converter.experimental_new_quantizer = True
#from tensorflow import keras
#from tensorflow import lite
def loss_fn(y_true, y_pred):
        return 1/np.log(2) * K.categorical_crossentropy(y_true, y_pred)
#model = keras.models.load_model(keras_file)
with open(args.param, 'r') as f:  
    params = json.load(f)
alphabet_size = len(params['id2char_dict'])+2 
model = getattr(models, 'LSTM_multi')(2048, int(args.sequence_length), alphabet_size)
model.load_weights(args.model)
#model = keras.models.load_model(keras_file,custom_objects={"loss_fn": loss_fn})
#begin=time()
#converter.experimental_new_quantizer = True
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter=tf.lite.TFLiteConverter.from_keras_model(keras_file)
#converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(keras_file)
#converter = lite.TFLiteConverter.from_keras_model_file(keras_file)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
#end=time()
#print(end-begin)
open(keras_file+'2048'+args.model, "wb").write(tflite_model)   # lite.h52048vertex200m.hdf5  后面会被删除
