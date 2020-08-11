from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
import skimage
import numpy as np
import os
from datetime import datetime

class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, tag, log_dir = './log'):
        now = datetime.now()
        super().__init__() 
        self.tag = tag
        self._train_run_name = 'path_image' + now.strftime("%H:%M:%S")
        self._log_writer_dir = log_dir
        
    def on_epoch_end(self, epoch, logs={}):
        decoder_weight = self.model.get_layer('decoder').get_weights()
        img = tf.nn.softmax(decoder_weight)
        img = tf.squeeze(img)
        img = tf.math.multiply(img, 10.0)
        shape = K.int_shape(img)
        if(len(shape) == 2):
            img = tf.reshape(img, [1, shape[0], shape[1], 1])
        elif(len(shape) == 1):
            img = tf.reshape(img, [1, 1, shape[0], 1])
            
        path = os.path.join(self._log_writer_dir, self._train_run_name)
        file_writer = tf.summary.create_file_writer(path)
        with file_writer.as_default():
            tf.summary.image("Decoder Paths", img, step=epoch, description='paths generated from softmaxing decoder weights')

        return

import time
    
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_batch_end(self, batch, logs=None):
        loss = logs.get('loss')
        d_weight = self.model.get_layer('decoder').get_weights()
        if tf.reduce_any(tf.math.is_nan(d_weight)):
            tf.print('loss is weird', loss)
        
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class Decoder_weights(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        dweight = self.model.get_layer('bottleneck').get_weights()
        print(dweight)
        
class Bottleneck_input_weights(keras.callbacks.Callback):
    def __init__(self, print_weights = False, log_dir = 'logs/'):
        self.p = print_weights
        self.fw = tf.summary.create_file_writer(log_dir)
        
    def on_epoch_end(self, epoch, logs={}):
        bweight = self.model.get_layer('bottleneck').get_weights()
        bweight_sum = K.sum(bweight[0], axis=0)
        self.fw.set_as_default()
        tf.summary.scalar('bottleneck weights', data=K.mean(bweight_sum), step=epoch)
        if self.p:
            print(bweight_sum)

class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            print("Early stopping requires %s available!" % self.monitor)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

