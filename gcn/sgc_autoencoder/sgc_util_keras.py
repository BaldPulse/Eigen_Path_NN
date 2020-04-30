from tensorflow import keras
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
                 bias_initializer='zeros',
                 **kwargs):
        self.output_dim = output_dim
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        super(sgc_decoder, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        super(sgc_decoder, self).build(input_shape)  # Be sure to call this at the end

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
        img = tf.squeeze(decoder_weight)
        #img = tf.nn.softmax(img)
        shape = K.int_shape(img)
        if(len(shape) == 2):
            img = tf.reshape(img, [1, shape[0], shape[1], 1])
        elif(len(shape) == 1):
            img = tf.reshape(img, [1, 1, shape[0], 1])
            
        path = os.path.join(self._log_writer_dir, self._train_run_name)
        file_writer = tf.summary.create_file_writer(path)
        with file_writer.as_default():
            tf.summary.image("Decoder Paths", img, step=0, description='paths generated from softmaxing decoder weights')

        return
