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


def get_sgc_model(num_nodes=41, num_sgc_feats=32, latent_size=1, all_paths = None):
    '''
    Input:
    num_nodes: Number of nodes in the network.
    num_sgc_feats: Number of features that the SGC outputs per node.
    latent_size: Size of the latent representation.
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
    
    latent = tf.keras.layers.Dense(latent_size,
                                   activation='relu',
                                   #kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None),
                                   kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.),
                                   kernel_regularizer = tf.keras.regularizers.l2(l=0.02),
                                   name='bottleneck',
                                   use_bias = True)(sgc_out)

    decode = sgc_decoder(num_nodes,
                         name = 'decoder',
                         kernel_constraint = clip_weights(),
                         kernel_regularizer = l1l2_corr_sm(l1 = 0.0, l2 = 0., kc = 0.0, ks = 0.01, kv = 0.0),
                         #kernel_regularizer = soft_binarize(0.0),
                         )(latent)
    if all_paths is not None:
        decode_weights = decode.get_weights()
        for i in range(len(all_paths)):
            for j in range(len(all_paths[i])):
                decode_weights[i,all_paths[i][j]] = 0.8
        decode.set_weights(decode_weights)
        decode = tf.keras.layers.Reshape((num_nodes, 1))(decode)
    
    model = tf.keras.Model(inputs=(I, A), outputs=[decode, latent])
    return model

def load_data(path):
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


learning_rate = 0.00008
batch_size = 32
n_epochs = 800

_log_dir = "/home/zhao/Documents/Eigen_Path_NN/gcn/sgc_autoencoder/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

file_writer = None

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=_log_dir, write_images=False, histogram_freq=1)
tbi_callback = TensorBoardImage('Image Example', log_dir = _log_dir)

current_path= os.getcwd()
print(current_path)
path = '/home/zhao/Documents/Eigen_Path_NN/gcn/data/simple_data/baseline_normalized_nl_0.2/'




_callbacks = [tensorboard_callback]
validation_prop = 0.0
_npath = -1
lthreshold = 0.1
uthreshold = 0.9
name = ''

flows, noiseless_flows, edge_adj, expected_paths = load_data(path)
flows = np.expand_dims(flows, 2)
noiseless_flows = np.expand_dims(noiseless_flows, 2)
A = prepare_adj(edge_adj)
A = np.tile(np.expand_dims(A, 0), (flows.shape[0], 1, 1))  # Adding a dummy batch dimension

n_edges = flows.shape[1]
n_sgc_feats = 32
n_flows = 5
n_paths = 1
_flows = flows

rearrange = 1
list_adj = adj_to_list(edge_adj)
identified_paths = None

while rearrange == 1:
    if n_paths == 1:
        model = get_sgc_model(n_edges, 
                              num_sgc_feats=n_sgc_feats, 
                              latent_size=n_paths)
    else:
        model = get_sgc_model(n_edges, 
                              num_sgc_feats=n_sgc_feats, 
                              latent_size=n_paths,
                              all_paths = identified_paths)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=[tf.keras.losses.MSE, None],
        metrics={'bottleneck': mean_pred}
    )


    model.fit(
        x={'input_node_features': _flows, 'adjacency': A},
        y= _flows,
        validation_split = 0.0,
        batch_size=batch_size,
        epochs=n_epochs,
        callbacks=_callbacks
    )
    weights = model.get_layer('decoder').get_weights()
    rigid_edges, soft_edges = identify_edges(weights[0])
    print(rigid_edges)
    identified_paths, _ =connect_edges(rigid_edges, soft_edges, edge_adj)
    print(identified_paths)
    file_writer = tf.summary.create_file_writer(_log_dir)
    paths = weights[0]
    l_shape = paths[0].shape
    l_img = np.reshape(paths, [1, n_paths, paths.shape[1], 1])
    l_img = tf.image.resize(l_img, [n_paths*5, paths.shape[1]*5 ])
    with file_writer.as_default():
        tf.summary.image("Learned paths\n", l_img, step=0)

    #find_linked_path(rigid_edges[0], list_adj)
    #print(rigid_edges)

'''
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
print(model.summary())
weights = model.get_layer('decoder').get_weights()
paths = weights[0]
now = datetime.now()
e_shape = expected_paths.shape
l_shape = paths[0].shape
l_img = np.reshape(paths, [1, n_paths, e_shape[1], 1])
l_img = tf.image.resize(l_img, [n_paths*5, e_shape[1]*5 ])
print(paths)
#print(weights[0])
epath = expected_paths
epath_sig = str(np.nonzero(epath))
#print(epath)
sim, vm, vs = evaluate_path_similarities(epath, binarize_paths(paths, lthres = lthreshold, uthres = uthreshold))
print("similarity index", sim)
file_writer = tf.summary.create_file_writer(_log_dir)
with file_writer.as_default():
    tf.summary.image("Learned paths\n" + epath_sig, l_img, step=0, description='Learned paths from decoder,'+ name + ', sim= %s' % sim + str(vm) + str(vs))




e_img = np.reshape(expected_paths, [1, e_shape[0], e_shape[1], 1])
e_img = tf.image.resize(e_img, [n_paths*5, e_shape[1]*5 ])
e_path = "logs/" +epath_sig +'/'
file_writer = tf.summary.create_file_writer(e_path)
with file_writer.as_default():
    tf.summary.image("Expected paths", e_img, step=0, description='expected paths to be learned')
'''
