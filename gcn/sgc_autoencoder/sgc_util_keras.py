from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
import skimage
import numpy as np
import os

class sgc_decoder(Layer):

    def __init__(self,
                 output_dim,
                 activation=None,                 
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 **kwargs):
        self.output_dim = output_dim
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        super(sgc_decoder, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        super(sgc_decoder, self).build(input_shape)  

    def call(self, x):
        kern = K.softmax(self.kernel)
        return K.dot(x, kern)
        #return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, tag, log_dir = './log'):
        super().__init__() 
        self.tag = tag
        self._train_run_name = 'path_image'
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

class l1l2_softmax(regularizers.Regularizer):
    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += self.l1 * K.sum(K.softmax(x))
        if self.l2:
            regularization += self.l2 * K.sum(K.square(K.softmax(x)))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2)}

class correlation(regularizers.Regularizer):
    def __init__(self, k = 0.01):
        self.k = K.cast_to_floatx(k)

    def __call__(self, x):
        regularization = 0.
        norm_sm_x = K.l2_normalize(K.softmax(x))
        corr = K.dot(norm_sm_x, K.transpose(norm_sm_x))
        for i in range(K.int_shape(corr)[0]):
            for j in range(K.int_shape(corr)[1]):
                if(j < i):
                    regularization += corr[i, j]
        return regularization

    def get_config(self):
        return {'k': float(self.k)}
    
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class l1l2_corr_sm(regularizers.Regularizer):
    def __init__(self, l1=0., l2=0., k = 0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.k = K.cast_to_floatx(k)

    def __call__(self, x):
        regularization = 0.
        sm_x = K.softmax(x)
        if self.l1:
            regularization += self.l1 * K.sum(sm_x)
        if self.l2:
            regularization += self.l2 * K.sum(K.square(sm_x))
        if self.k:
            norm_sm_x = K.l2_normalize(sm_x)
            corr = K.dot(norm_sm_x, K.transpose(norm_sm_x))
            for i in range(K.int_shape(corr)[0]):
                for j in range(K.int_shape(corr)[1]):
                    if(j < i):
                        regularization += corr[i, j]
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2),
                'k' : float(self.k)}
