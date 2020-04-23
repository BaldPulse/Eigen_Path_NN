import os
# This disables my GPU.
os.environ["CUDA_VISIBLE_DEVICES"]=""

import tensorflow as tf
import numpy as np

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
    
    sgc_out = sgc_layer(I, A, featsize=num_sgc_feats, numhop=3)  # Shape: [num_nodes, num_sgc_feats]
    sgc_out = tf.keras.layers.ReLU()(sgc_out)
    sgc_out = tf.keras.layers.Flatten()(sgc_out)  # Shape: [num_nodes*num_sgc_feats]
    
    latent = tf.keras.layers.Dense(latent_size, activation='relu', name='bottleneck')(sgc_out)
    
    # Consider adding `kernel_regularizer=tf.keras.regularizers.l1()` to below,
    # to encourage sparser (mostly zero) weights for the decoder.
    decode = tf.keras.layers.Dense(num_nodes, use_bias=False, name='decoder')(latent)
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



path = '../data/5/'
flows, noiseless_flows, edge_adj, expected_paths = load_data(path)
flows = np.expand_dims(flows, 2)
noiseless_flows = np.expand_dims(noiseless_flows, 2)
A = prepare_adj(edge_adj)
A = np.tile(np.expand_dims(A, 0), (flows.shape[0], 1, 1))  # Adding a dummy batch dimension

n_edges = flows.shape[1]
n_sgc_feats = 2
n_flows = 5
n_paths = 2

learning_rate = 0.0003
batch_size = 32
n_epochs = 200

model = get_sgc_model(n_edges, 
                      num_sgc_feats=n_sgc_feats, 
                      latent_size=n_paths)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.MSE,
    # You could add metrics to track during training, 
    # e.g. difference between the decoder weight and ground truth paths.
)

model.fit(
    x={'input_node_features': flows, 'adjacency': A},
    y=noiseless_flows,
    batch_size=batch_size,
    epochs=n_epochs,
    # validation data=(x={F, A}, Y=F), to include validation data here, 
    # you could also include logging to a file via callbacks,
    # saving model to a file via callbacks, etc 
)

model.save(path+'sgc.h5')

print(model.summary())
paths = model.get_layer('decoder').get_weights()[0]
print(paths) # Decoder weights
print(paths.shape)
if(not expected_paths is None):
    print(expected_paths.shape)
    print(expected_paths)
    w = expected_paths @ paths.T/(paths @ paths.T)
    print((w @ paths).astype('int'))
# To predict after fitting the model:
#model.predict(x={foobar})
