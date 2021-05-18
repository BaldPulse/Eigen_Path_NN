import tensorflow as tf
import numpy as np
import numpy.linalg as npla
'''
legacy code for gcn
'''


def prepare_adj(adj):
    '''
    turns adjacency matrix into gcn propagation matrix
    input:
        adj: adjacency matrix for graph
    output:
        adj_w: weights for propagation
    '''
    print("adjacency matrix (edge)", adj[0:5])
    I = np.identity(adj.shape[0])
    adj_bar = adj + I
    D = np.sum(adj_bar, axis = 1)
    return np.diag(D**(-0.5)) @ adj_bar @ np.diag(D**(-0.5))


def sgc_layer(H, A, featsize, numhop):
    layer = tf.matmul(A, H)
    for i in range(numhop - 1):
        layer = tf.matmul(A, layer)
    layer = tf.keras.layers.Dense(featsize)(layer)
    return layer

def get_gcn_model(num_nodes=41, num_sgc_feats=2):
    '''
    Input:
    num_nodes: Number of nodes in the network.
    num_sgc_feats: Number of features that the SGC outputs per node.
    ----
    Returns: A Keras model which takes as input `input_node_features` and `adjacency` and outputs
        a matrix of the same shape as `input_node_features`. 
    '''
    I = tf.keras.Input((num_nodes, 1), name='input_node_features')
    A = tf.keras.Input((num_nodes, num_nodes), name='adjacency')
    
    sgc_out = sgc_layer(I, A, featsize=num_sgc_feats, numhop=4)  # Shape: [num_nodes, num_sgc_feats]
    sgc_out = tf.keras.layers.ReLU()(sgc_out)
    sgc_out = tf.keras.layers.concatenate([sgc_out, I], axis=-1)
    sgc_out = tf.keras.layers.Flatten()(sgc_out)  # Shape: [num_nodes*num_sgc_feats]
    
    
    model = tf.keras.Model(inputs=(I, A), outputs=[decode, latent])
    return model