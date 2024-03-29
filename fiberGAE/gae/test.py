from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import pickle

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.input_data import load_data
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.') # default 0.01
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.') # default 200
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.') # default 32
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.') # default 16
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_string('datapath', '/data/hyf/atlas/train/H_4/1000/', 'load data from')

model_str = FLAGS.model
dataset_str = FLAGS.dataset
datapath_str = FLAGS.datapath


# Store original adjacency matrix (without diagonal entries) for later

loadpath = '/data/hyf/swm/model1_data/test_5subjects.pickle'
# loadpath = '/data/hyf/swm/model1_data/train_range2.pickle'

with open(loadpath, 'rb') as f:
    datasets = pickle.load(f)

adj = datasets['adj']
features = datasets['features']

for i in range(len(adj)):
    print(np.count_nonzero(adj[i]), adj[i].shape)

adj_norm = []
adj_orig = []
adj_train = []
train_edges = []
val_edges = []
test_edges = []
val_edges_false = []
test_edges_false = []
for i in range(len(adj)):
    print(adj[i].shape, features[i].shape)
    adj[i] = sp.csr_matrix(adj[i])
    features[i] = sp.csr_matrix(features[i])
    adj_orig.append(adj[i])
    adj_orig[i] = adj_orig[i] - sp.dia_matrix((adj_orig[i].diagonal()[np.newaxis, :], [0]), shape=adj_orig[i].shape)
    adj_orig[i].eliminate_zeros()

    adj_train_single, train_edges_single, val_edges_single, val_edges_false_single, test_edges_single, test_edges_false_single = mask_test_edges(adj[i])
    train_edges.append(train_edges_single)
    val_edges.append(val_edges_single)
    test_edges.append(test_edges_single)
    val_edges_false.append(val_edges_false_single)
    test_edges_false.append(test_edges_false_single)

    adj[i] = adj_train_single
    adj_train.append(adj_train_single)

    if FLAGS.features == 0:
        features[i] = sp.identity(features[i].shape[0])  # featureless

    # Some preprocessing
    adj_norm.append(preprocess_graph(adj[i]))

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

num_nodes = adj[0].shape[0]

for i in range(len(features)):
    features[i] = sparse_to_tuple(features[i].tocoo())

num_features = features[0][2][1]
features_nonzero = features[0][1].shape[0]

# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

pos_weight = float(adj[0].shape[0] * adj[0].shape[0] - adj[0].sum()) / adj[0].sum()
norm = adj[0].shape[0] * adj[0].shape[0] / float((adj[0].shape[0] * adj[0].shape[0] - adj[0].sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []


def get_roc_score(edges_pos, edges_neg, adj_orig, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


cost_val = []
acc_val = []
val_roc_score = []

adj_label = []
for i in range(len(adj_train)):
    adj_label.append(adj_train[i] + sp.eye(adj_train[i].shape[0]))
    adj_label[i] = sparse_to_tuple(adj_label[i])

saver = tf.train.Saver()
saver.restore(sess, "/data/hyf/swm/model1_data/gae_models/model.ckpt")

embeddings = []
for i in range(len(adj)):
    feed_dict = construct_feed_dict(adj_norm[i], adj_label[i], features[i], placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    
    outs = sess.run([opt.cost, opt.accuracy, model.z_mean], feed_dict=feed_dict)

    embeddings.append(outs[-1])

print(len(embeddings))
savepath = "/data/hyf/swm/model1_data/embeddings/embeddings_test.pkl"

with open(savepath, 'wb') as file:
     pickle.dump(embeddings, file)
print("saved embeddings!")
