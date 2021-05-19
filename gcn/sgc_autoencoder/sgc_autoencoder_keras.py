import os
# This disables my GPU.
#os.environ["CUDA_VISIBLE_DEVICES"]=""
import sys, getopt
optlist, args = getopt.getopt(sys.argv[1:], 'n:p:h', ['noiseless', 'data=', 'earlystop', 'tbi', 'tensorboard', 'mixed=', 'printdweight', 'cpu', 'logbweight=', 'lowerthreshold=', 'upperthreshold=', 'name='])
for a,o in optlist:
    if a=='--cpu':        
        # This disables my GPU.
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    
import tensorflow as tf
import numpy as np
import numpy.linalg as npla

from graph import Network, Network_flows, Propagation
from sgc_layer_utils import sgc_decoder, l1l2_softmax, correlation, l1l2_corr_sm, soft_binarize, clip_weights, mean_pred
from sgc_callbacks import TensorBoardImage, datetime
from sgc_path_utils import binarize_paths, weighted_jaccard, jaccard_multiset, evaluate_path_similarities, l1_normalize
from join_paths import identify_edges, find_linked_path, adj_to_list, connect_edges

def prepare_adj(adj):
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


def get_sgc_model(G, sources, num_sgc_feats=32, latent_size=1):
    '''
    Input:
    G: a representation of the network
    num_sgc_feats: Number of features that the SGC outputs per node(edge).
    latent_size: Size of the latent representation.
    ----
    Returns: A Keras model which takes as input `input_node_features` and `adjacency` and outputs
        a matrix of the same shape as `input_node_features`. 
        also allows for output of latent layer for uses in propagation loss
    '''
    I = tf.keras.Input((G.nNodes, 1), name='input_node_features')
    A = tf.keras.Input((G.nNodes, G.nNodes), name='adjacency')
    
    sgc_out = sgc_layer(I, A, featsize=num_sgc_feats, numhop=4)  # Shape: [num_nodes, num_sgc_feats]
    sgc_out = tf.keras.layers.ReLU()(sgc_out)
    sgc_out = tf.keras.layers.concatenate([sgc_out, I], axis=-1)
    sgc_out = tf.keras.layers.Flatten()(sgc_out)  # Shape: [num_nodes*num_sgc_feats]
    
    latent = tf.keras.layers.Dense(latent_size,
                                   activation='relu',
                                   #kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None),
                                   kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.),
                                   kernel_regularizer = tf.keras.regularizers.l2(l=0.02),
                                   name='bottleneck',
                                   use_bias = True)(sgc_out)

    decode_layer = sgc_decoder(G.nNodes,
                         name = 'decoder',
                         kernel_constraint = clip_weights(),
                         kernel_regularizer = l1l2_corr_sm(l1 = 0.0, l2 = 0., kc = 0.0, ks = 0.01, kv = 0.0),
                         #kernel_regularizer = soft_binarize(0.0),
                         Edges = G.Edges,
                         nNodes = G.nNodes,
                         sources = sources,
                         )
    decode = decode_layer(latent)
    
    model = tf.keras.Model(inputs=(I, A), outputs=decode)
    return model

def set_weights(model, all_paths):
    '''
    initialize/set the weight of a keras model by the paths provided. The dimensions of the paths must correspond to the dimensions of the model
    '''
    decode_layer = model.get_layer('decoder')
    decode_weights = decode_layer.get_weights()[0]
    latent_layer = model.get_layer('bottleneck')
    latent_weights = latent_layer.get_weights()
    print("decode_weights", decode_weights[0])
    print("latent_weights", latent_weights[0])
    for i in range(len(all_paths)):
        for j in range(len(all_paths[i])):
            decode_weights[i,all_paths[i][j]] = 0.8
    decode_layer.set_weights([decode_weights])

def load_data(path):
    '''
    load synthetic data produced by the jupyter notebook in the data directory
    '''
    flows = np.load(os.path.join(path, 'flows.npy'))
    edge_adj = np.load(os.path.join(path, 'adj.npy'))
    try:
        expected_paths = np.load(os.path.join(path, 'paths.npy'))
    except IOError:
        expected_paths = None
        print('no expected paths in'+path)
    try:
        noiseless_flows = np.load(os.path.join(path, 'noiseless_flows.npy'))
    except IOError:
        noiseless_flows = None
        print('no noiseless flows in'+path)
    return flows, noiseless_flows, edge_adj, expected_paths


n_edges = flows.shape[1]
n_sgc_feats = 32
n_flows = 5
n_paths = 1
_flows = flows

learning_rate = 0.00008
batch_size = 32
n_epochs = 400

_log_dir = "/home/zhao/Documents/Eigen_Path_NN/gcn/sgc_autoencoder/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

file_writer = None

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=_log_dir, write_images=False, histogram_freq=1)
tbi_callback = TensorBoardImage('Image Example', log_dir = _log_dir)

current_path= os.getcwd()
print(current_path)
path = '/home/zhao/Documents/Eigen_Path_NN/gcn/data/simple_data/baseline_normalized_nl_0.2/'

model = get_sgc_model(n_edges, 
                            num_sgc_feats=n_sgc_feats, 
                            latent_size=n_paths)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=[tf.keras.losses.MSE, None],
    metrics={'bottleneck': mean_pred}
)
print("weights", model.get_layer("bottleneck").get_weights())
if n_paths != 1:
    set_weights(model, identified_paths)
print(type(n_epochs))
model.fit(
    x={'input_node_features': _flows, 'adjacency': A},
    y= _flows,
    validation_split = 0.0,
    batch_size=batch_size,
    epochs=n_epochs,
    callbacks=_callbacks
)