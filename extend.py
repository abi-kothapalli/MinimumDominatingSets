from __future__ import division
from __future__ import print_function

import os
import time
from statistics import median
import networkx as nx
import json

import scipy.io as sio
import numpy as np
import tensorflow as tf
from gcn.utils import *
from gcn.models import GCN_DEEP_DIVER
from graph_methods import *
from iterative_greedy import iterativeGreedy, IG_GCN

RUN_NAME = "FINAL_RUN"
N_bd = 32

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 250, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('diver_num', 32, 'Number of outputs.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probaNUmbility).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 1, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('num_layer', 20, 'number of layers.')

# Some preprocessing
num_supports = 1 + FLAGS.max_degree
model_func = GCN_DEEP_DIVER

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=(None, N_bd)), # featureless: #points
    'labels': tf.placeholder(tf.float32, shape=(None, 2)), # 0: not linked, 1:linked
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=N_bd, logging=True)

# use gpu 0
os.environ['CUDA_VISIBLE_DEVICES']=str(0)

# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Init variables
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

ckpt=tf.train.get_checkpoint_state(f"{RUN_NAME}")
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

# Define model evaluation function
def testingEvaluataion(features, support, placeholders):
    feed_dict_val = construct_feed_dict4pred(features, support, placeholders)
    outs_val = sess.run([model.outputs_softmax], feed_dict=feed_dict_val)
    return outs_val[0]

testing_analysis = {}

graph_model = 0 # Use 0 for binomial (Erdos-Renyi) or 1 for scale-free (Barabasi-Albert)
output_filename = "test-results.json"

for graph_size in np.arange(250, 1001, 10):
    for edge_prob in [0.01, 0.015, 0.02]:

        if graph_model == 0:
            g = nx.erdos_renyi_graph(graph_size, edge_prob)
        else:
            g = nx.barabasi_albert_graph(graph_size, int((graph_size - 1) * edge_prob / 2))

        adj = nx.adjacency_matrix(g).todense()

        print(f"Generating greedy/random solution for graph of {graph_size} nodes and density {edge_prob}")
        
        greedySize, greedyTime = greedySolution(g)
        randomSize, randomTime = randomSolution(g)

        print(f"Getting GCN solutions")

        startTime = time.time()
        nn = adj.shape[0]
        features = np.ones([nn, N_bd])
        features = sp.lil_matrix(features)
        features = preprocess_features(features)
        support = simple_polynomials(adj, FLAGS.max_degree)

        outs = testingEvaluataion(features, support, placeholders)

        sol, solution_sizes, avgTime = getBestMDS(g, outs)
        runtime = time.time() - startTime
        print(f"Found GCN solutions")

        IGSize, IGTime = iterativeGreedy(g)

        IGGCNSize, IGGCNTime = IG_GCN(g, outs.transpose())

        testing_analysis[f"{graph_size}_{edge_prob}"] = {
            'best_gcn': len(sol),
            'iterative_greedy': len(IGSize),
            'ig_gcn': IGGCNSize,
            'gcn_solutions': solution_sizes,
            'greedy': greedySize,
            'random': randomSize,
            'gcn_runtime_total': runtime,
            'gcn_runtime_per_prediction': avgTime,
            'iterative_greedy_time': IGTime,
            'ig_gcn_time': IGGCNTime,
            'greedy_time': greedyTime,
            'random_time': randomTime,
        }

        with open(output_filename, "w") as f:
            json.dump(testing_analysis, f, indent=2)