import tensorflow as tf
import numpy as np
import numpy.linalg as npla
import sys


def prepare_adj(adj):
    I = np.identity(adj.shape[0])
    adj_bar = adj + I
    D = np.sum(adj_bar, axis = 1)
    return np.diag(D**(-0.5)) @ adj_bar @ np.diag(D**(-0.5))

def sgc_layer(numhop, W, A, H):
    layer = tf.matmul(A, H)
    for i in range(numhop - 1):
        layer = tf.matmul(A, layer)
    layer = tf.matmul(layer, W)
    return layer

def sgc_autoencoder(X, We, Wd, A):
    buff = []
    X_unpacked = tf.unstack(X)
    for x in X_unpacked:
        E = sgc_layer(numhop=4, W = We, A=A, H=tf.transpose(tf.expand_dims(x,0)))
        A_inv = tf.linalg.inv(A)
        x_bar = sgc_layer(numhop=4, W=Wd, A=A_inv, H=E)
        buff.append(tf.transpose(x_bar))
    return tf.concat(buff, 0)

def load_data(path):
    flows = np.load(path+'flows.npy')
    edge_adj = np.load(path+'adj.npy')
    return flows, edge_adj

path = '/home/zhaotang/Documents/eigen_path/gcn/data/1/'
n_features = 1
n_paths    = 2

flows, edge_adj = load_data(path)
n_edges = flows.shape[1]

weights = {
    'encoder': tf.Variable(tf.random_normal([n_features, n_paths])),
    'decoder': tf.Variable(tf.random_normal([n_paths, n_features]))
}

training_epochs = 300
batch_size = 32
dropout = tf.placeholder("float")

X = tf.placeholder("float", [batch_size, n_edges])
A = tf.constant(prepare_adj(edge_adj), dtype = "float" , shape = [n_edges, n_edges])

pred_flows = sgc_autoencoder(X, weights['encoder'], weights['decoder'], A)
loss       = tf.nn.l2_loss(tf.math.subtract(pred_flows, X))
optimizer  = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
We_print   = tf.print("encoder:", weights['encoder'], output_stream=sys.stdout)
Wd_print   = tf.print("decoder:", weights['decoder'], output_stream=sys.stdout)

path_val = '/home/zhaotang/Documents/eigen_path/gcn/data/2/'

flows_val, edge_adj_val = load_data(path_val)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_loss = 0.0
        total_batch = int(len(flows) / batch_size)
        for i in range(total_batch-1):
            batch_X = flows[i*32:(i+1)*32]
            _, c = sess.run([optimizer, loss], 
                            feed_dict={
                                X: batch_X,
                                dropout: 0.2
                            })
            avg_loss += c / total_batch
        print("Epoch:", '%04d' % (epoch+1), "loss=", \
                "{:.9f}".format(avg_loss))
        if(avg_loss < 0.5):
            break
    sess.run([We_print, Wd_print])
    
    avg_loss_val = 0.0
    total_batch_val = int(len(flows_val) / batch_size)
    for i in range(total_batch_val-1):
        batch_X_val = flows_val[i*32:(i+1)*32]
        [c] = sess.run([loss],
                     feed_dict={
                         X: batch_X_val,
                         dropout: 0.2
                     })
        avg_loss_val += c / total_batch_val
    print("Validation loss:", avg_loss_val)
