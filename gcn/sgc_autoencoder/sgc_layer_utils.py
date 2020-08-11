from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import math_ops
import tensorflow as tf
import skimage
import numpy as np
import os
from datetime import datetime

class sgc_decoder(Layer):

    def __init__(self,
                 output_dim,
                 activation=None,                 
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        self.output_dim = output_dim
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        super(sgc_decoder, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        super(sgc_decoder, self).build(input_shape)

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


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
    

class l1l2_corr_sm(regularizers.Regularizer):
    def __init__(self, l1=0., l2=0., kc = 0., ks = 0., kv = 0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.kc = K.cast_to_floatx(kc)
        self.ks = K.cast_to_floatx(ks)
        self.kv = K.cast_to_floatx(kv)
        
    @tf.function
    def __call__(self, x):
        regularization = 0.
        sig_x = x#K.sigmoid(x*5)
        if tf.math.is_nan(x)[0,0]:
            tf.print('x is nan', x)
        #l1 regularization for individual paths
        #incentivize paths to be sparse
        if self.l1:
            regularization += self.l1 * K.sum(K.abs(x))
        if self.l2:
            regularization += self.l2 * K.sum(K.square(sig_x))
        #incentivize paths to be different
        if self.kc:
            norm_sig_x = K.l2_normalize(sig_x)
            corr = K.dot(norm_sig_x, K.transpose(norm_sig_x))
            for i in range(K.int_shape(corr)[0]):
                for j in range(K.int_shape(corr)[1]):
                    if(j < i):
                        regularization += self.kc * corr[i, j]
        #incentivize paths to be similar in sparsity
        if self.ks:
            sig_x_std = K.var(sig_x, axis = -1)
            std = K.var(sig_x_std)
            regularization += std
        if self.kv:
            sig_x_std = K.var(sig_x, axis = -1)
            regularization -= K.sum(sig_x_std)
        return regularization
    
    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2),
                'k' : float(self.kc)}

class soft_binarize(regularizers.Regularizer):
    def __init__(self, k):
        self.k = k
        
    def __call__(self, x):
        reg_0 = math_ops.reduce_sum(math_ops.square(x))
        reg_1 = math_ops.reduce_sum(math_ops.square(x-1))
        return self.k * K.minimum(reg_0, reg_1)
        #return self.k * reg_1

class clip_weights(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, upper = 1.0, lower = 0.0):
        self.upper = upper
        self.lower = lower

    def __call__(self, p):
        return K.clip(p, self.lower, self.upper)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'upper': self.upper,
                'lower': self.lower}

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
    
def dummy_loss(y_true, y_pred):
    return 0
