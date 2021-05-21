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

        for key, value in kwargs.items():
            if(key is "Edges"):
                self.Edges = value
                self.nEdges = len(value)
            if(key is "nNodes"):
                self.nNodes = value
            if(key is "sources"):
                self.sources = value
                # print("self.sources", self.sources)
        #create a source termplate tensor where the value of source nodes are 1
        self.source_flow = tf.zeros([self.nNodes], tf.float32)
        for s in self.sources:
                self.source_flow[s] = 1.0
        
        super(sgc_decoder, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.nEdges),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        super(sgc_decoder, self).build(input_shape)

    def call(self, x):
        '''
        multiple output decoder layer with two reconstructed flows:
            output[0]: reconstruction based on propagation
            output[1]: reconstruction based on linear combination
        '''
        factors = x
        paths  = self.kernel
        node_prop = tf.zeros([tf.shape(paths)[0],self.nNodes, self.nNodes], tf.float32)
        for i in range(tf.shape(paths)[0]):
            for e in range(nEdges):
                node_prop[i,self.Edges[e][1], self.Edges[e][0]] = paths[i][e]
        node_prop[:, 0, :] = 1-K.sum(node_prop, axis=1)
        
        prop_flows = tf.zeros([tf.shape(factors)[0], self.nNodes])
        flow_count = 0
        for factor in factors:
            prop_flow = tf.zeros([factor.shape[0], self.nNodes]) # # of paths * # of edges
            for i in range(len(factor)):
                prop_flow[i] = self.source_flow[i] * factor[i] #set every source to the path factor
                iter = tf.math.count_nonzero(paths[i])
                for j in range(iter):
                    prop_flow[i] = K.dot(node_prop[i], prop_flow[i].T).T
                for s in self.sources:
                    prop_flow[i, s] -= factor[i] #reset source by path factor to account for outgoing flows
            prop_flows[flow_count] = K.sum(prop_flow, axis=0)
            flow_count+=1

        return [prop_flows, K.dot(x, paths)]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.nNodes), (input_shape[0], self.nEdges)]


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
