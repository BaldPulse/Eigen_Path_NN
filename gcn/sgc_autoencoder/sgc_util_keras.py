from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
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
        kern = K.sigmoid(5*self.kernel)
        return K.dot(x, kern)
        #return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


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

    def on_batch_end(self, batch, logs=None):
        loss = logs.get('loss')
        d_weight = self.model.get_layer('decoder').get_weights()
        if tf.reduce_any(tf.math.is_nan(d_weight)):
            tf.print('loss is weird', loss)
        
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class Bottleneck_input_weights(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        bweight = self.model.get_layer('bottleneck').get_weights()
        bweight_sum = K.sum(bweight[0], axis=0)
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
        sig_x = K.sigmoid(x*5)
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


def weighted_jaccard(a, b):
    a_pad = a
    b_pad = b
    #pad
    if b.shape[0] > a.shape[0]:
        N = b.shape[0] - a.shape[0]
        a_pad = np.pad(a, (0,N), 'constant')
    elif a.shape[0] > b.shape[0]:
        N = a.shape[0] - b.shape[0]
        b_pad = np.pad(b, (0,N), 'constant')
    return (2-np.sum(np.absolute(a_pad-b_pad)))/(2+np.sum(np.absolute(a_pad-b_pad)))


def jaccard_multiset(s, m):
    vm = np.zeros(s.shape[0])
    vs = np.ones(s.shape[0])
    for learned_path in m:
        max_sim = -1
        max_i = -1
        for i in range(s.shape[0]):
            sim = weighted_jaccard(learned_path, s[i])
            if sim>max_sim:
                max_sim = sim
                max_i = i
        vm[max_i] += max_sim
    print(vm, vs)
    return weighted_jaccard(vm, vs)


def evaluate_path_similarities(epath, lpath):
    return jaccard_multiset(epath, lpath)

def l1_normalize(A):
    return A / np.sum(np.absolute(A), axis=1).reshape(A.shape[0],1)

if __name__ == "__main__":
    t0 = np.array([[1,2,3,1],
                   [2,0,0,1]])
    t0 = t0 / np.sum(np.absolute(t0), axis=1).reshape(t0.shape[0],1)
    t1 = np.array([[1,2,3,0],
                   [1,0,0,1]])
    t1 = t1 / np.sum(np.absolute(t1), axis=1).reshape(t1.shape[0],1)
    #print(t0, t0,weighted_jaccard(t0, t0))
    #print(t0, t1,weighted_jaccard(t0, t1))
    print(t0, t1)
    print(jaccard_multiset(t0, t1))
    #print(weighted_jaccard([1,2,3,1], [1,2,3,0]))
    #print(weighted_jaccard([1,2,3,1], [2,0,0,1]))
    #print(weighted_jaccard([1,0,0,1], [2,0,0,1]))
