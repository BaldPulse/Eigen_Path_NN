import os
# This disables my GPU.
#os.environ["CUDA_VISIBLE_DEVICES"]=""

import tensorflow as tf
import numpy as np
import numpy.linalg as npla
import sys, getopt
from sgc_layer_utils import *
from sgc_callbacks import *


def prepare_adj(adj):
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


def get_sgc_model(num_nodes=41, num_sgc_feats=32, latent_size=1):
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
    sgc_out = tf.keras.layers.Flatten()(sgc_out)  # Shape: [num_nodes*num_sgc_feats]
    
    latent = tf.keras.layers.Dense(latent_size,
                                   activation='relu',
                                   #kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None),
                                   kernel_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.),
                                   name='bottleneck')(sgc_out)

    decode = sgc_decoder(num_nodes,
                         name = 'decoder',
                         kernel_constraint = clip_weights(),
                         kernel_regularizer = l1l2_corr_sm(l1 = 0.0, l2 = 0., kc = 0.0, ks = 0.4, kv = 0.0),
                         #kernel_regularizer = soft_binarize(0.0),
                         )(latent)
    decode = tf.keras.layers.Reshape((num_nodes, 1))(decode)
    
    model = tf.keras.Model(inputs=(I, A), outputs=decode)
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

def create_profusion(pure0, pure1, mixed, mixed_portion):
    print('pure0', pure0.shape)
    total = 2000
    mixed_num = int(total * mixed_portion)
    pure_num = int((total - mixed_num)/2)
    #print(np.random.choice(pure0.shape[0], size=pure_num, replace=False))
    sample = np.append(pure0[np.random.choice(pure0.shape[0], size=pure_num, replace=False)],\
                       mixed[np.random.choice(mixed.shape[0], size=mixed_num, replace=False)], axis=0)
    sample = np.append(pure1[np.random.choice(pure1.shape[0], size=pure_num, replace=False)], sample, axis=0)
    return sample

optlist, args = getopt.getopt(sys.argv[1:], 'n:p:', ['noiseless', 'data=', 'earlystop', 'tbi', 'tensorboard', 'mixed=', 'printdweight'])

learning_rate = 0.0003
batch_size = 32
n_epochs = 300

_log_dir = './logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=_log_dir, write_images=False, histogram_freq=1)
tbi_callback = TensorBoardImage('Image Example', log_dir = _log_dir)
earlystop_callback = EarlyStoppingByLossVal(monitor='loss', value=3, verbose=0)
biw_callback = Bottleneck_input_weights()
dw_callback = Decoder_weights()
time_callback = TimeHistory()

path = '../data/baseline/'




_callbacks = []
use_noiseless = False
use_profusion = False
mixed_proportion = 0
_npath = -1
for a,o in optlist:
    if a=="-n":
        n_epochs=int(o)
    if a=='--noiseless':
        use_noiseless = True
    if a=='--mixed':
        use_profusion = True
        print('mixed proportion', o)
        mixed_proportion = float(o)
    if a=='--data':
        path='../data/'+o
    if a=='--tensorboard':
        _callbacks.append(tensorboard_callback)
    if a=='--tbi':
        _callbacks.append(tbi_callback)
    if a=='--earlystop':
        _callbacks.append(earlystop_callback)
    if a=='--printdweight':
        _callbacks.append(dw_callback)
    if a=='-p':
        _npath = int(o)

flows, noiseless_flows, edge_adj, expected_paths = load_data(path)
flows = np.expand_dims(flows, 2)
noiseless_flows = np.expand_dims(noiseless_flows, 2)
A = prepare_adj(edge_adj)
A = np.tile(np.expand_dims(A, 0), (flows.shape[0], 1, 1))  # Adding a dummy batch dimension

if use_profusion:
    mixed_path = '../data/baseline_mixed/'
    m_flows, m_noiseless_flows, edge_adj, expected_paths = load_data(mixed_path)
    pure0_path = '../data/baseline_pure0/'
    p0_flows, p0_noiseless_flows, edge_adj, p0_expected_paths = load_data(pure0_path)
    pure1_path = '../data/baseline_pure1/'
    p1_flows, p1_noiseless_flows, edge_adj, p1_expected_paths = load_data(pure1_path)
    flows = create_profusion(p0_flows, p1_flows, m_flows, mixed_proportion)
    noiseless_flows = create_profusion(p0_noiseless_flows, p1_noiseless_flows, m_noiseless_flows, mixed_proportion)

n_edges = flows.shape[1]
n_sgc_feats = 32
n_flows = 5
n_paths = expected_paths.shape[0]
_flows = flows
if use_noiseless:
    _flows = noiseless_flows
if _npath >0:
    n_paths = _npath


model = get_sgc_model(n_edges, 
                      num_sgc_feats=n_sgc_feats, 
                      latent_size=n_paths)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.MSE,
)


model.fit(
    x={'input_node_features': _flows, 'adjacency': A},
    y= _flows,
    batch_size=batch_size,
    epochs=n_epochs,
    callbacks=_callbacks
)



e_shape = expected_paths.shape
e_img = np.reshape(expected_paths, [1, e_shape[0], e_shape[1], 1])
e_path = os.path.join(_log_dir, 'e_paths')
file_writer = tf.summary.create_file_writer(e_path)
with file_writer.as_default():
    tf.summary.image("Expected paths", e_img, step=0, description='expected paths to be learned')
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
print(model.summary())
weights = model.get_layer('decoder').get_weights()
paths = weights[0] * 5
now = datetime.now()
l_shape = paths[0].shape
l_img = np.reshape(paths, [1, n_paths, e_shape[1], 1])
l_path = os.path.join(_log_dir, now.strftime("%H:%M:%S"))
file_writer = tf.summary.create_file_writer(l_path)
with file_writer.as_default():
    tf.summary.image("Learned paths", l_img, step=0, description='Learned paths from decoder')
print(paths)
#print(weights[0])
l1_epath = expected_paths
#print(l1_epath)
#sim = evaluate_path_similarities(l1_epath, paths.numpy())
#print(sim)

# To predict after fitting the model:
#model.predict(x={foobar})
