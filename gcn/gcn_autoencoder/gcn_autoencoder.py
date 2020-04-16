import tensorflow as tf
import numpy as np

def gcn_output(A, H, W):
    layer = tf.matmul(H, W)
    layer = tf.matmul(A, layer)#H_out = AH_lW
    return layer

def gcn_hidden(A, H, W, dropout):
    print("input H:\n",H)
    layer = tf.matmul(H, W)
    print("layer:\n",layer)
    layer = tf.matmul(A, layer) 
    layer = tf.nn.relu(layer) #H_l = relu(AH_(l-1)W)
    layer_output = tf.nn.dropout(layer, rate = dropout)
    return layer_output

def path_decoder(P, W):
    layer = tf.matmul(P, W)
    return layer

def prepare_adj(adj):
    I = np.identity(adj.shape[0])
    adj_bar = adj + I
    D = np.sum(adj_bar, axis = 1)
    return np.diag(D**(-0.5)) @ adj_bar @ np.diag(D**(-0.5))

def gcn_atuoencoder(X, W, A):
    pred_flows = []
    for i, x in enumerate(X):
        H1 = gcn_hidden(A, x.T, W['g1'], dropout)
        H2 = gcn_hidden(A, H1,W['g2'], dropout)
        H3 = gcn_hidden(A, H2,W['g3'], dropout)
        P  = gcn_output(A, H3,W['g4'])

        pred_flows += path_decoder(P, W['out'])
    return tf.concat

def load_data(path):
    flows = np.load(path+'flows.npy')
    edge_adj = np.load(path+'adj.npy')
    return flows, edge_adj



path = '/home/zhaotang/Documents/eigen_path/gcn/data/'

flows, edge_adj = load_data(path)
n_edges = flows.shape[1]

n_feature_0 = 1
n_feature_1 = 7
n_feature_2 = 5
n_feature_3 = 3

n_paths = 2

weights = {
    'g1': tf.Variable(tf.random_normal([n_feature_0, n_feature_1])),
    'g2': tf.Variable(tf.random_normal([n_feature_1, n_feature_2])),
    'g3': tf.Variable(tf.random_normal([n_feature_2, n_feature_3])),
    'g4': tf.Variable(tf.random_normal([n_feature_3, n_paths])),
    
    'out': tf.Variable(tf.random_normal([n_paths, n_edges]))    
}

dropout = tf.placeholder("float")

training_epochs = 1000
batch_size = 32

X = tf.placeholder("float", [None, n_edges])
A = tf.constant(prepare_adj(edge_adj), dtype = "float" , shape = [n_edges, n_edges])

pred_flows = gcn_atuoencoder(X, weights, A)
loss = tf.nn.l2_loss(tf.math.subtract(pred_flows, X))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(len(flows) / batch_size)
        flows_batches = np.array_split(flows, total_batch)
        for i in range(total_batch):
            batch_X = flows_batches[i]
            _, c = sess.run([optimizer, loss], 
                            feed_dict={
                                X: batch_X,
                                dropout: 0.2
                            })
            avg_cost += c / total_batch
        print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
