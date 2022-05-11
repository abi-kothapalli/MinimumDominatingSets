from __future__ import division
from __future__ import print_function

import os
import time
from statistics import median
import json

import scipy.io as sio
import numpy as np
import tensorflow as tf
from gcn.utils import *
from gcn.models import GCN_DEEP_DIVER

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
    t_test = time.time()
    feed_dict_val = construct_feed_dict4pred(features, support, placeholders)
    outs_val = sess.run([model.outputs_softmax], feed_dict=feed_dict_val)
    return (time.time() - t_test), outs_val[0]

def getBestMDS(adj, predictions):
    start = time.time()
    g = nx.from_numpy_matrix(adj)

    bestSolution = list(range(adj.shape[0]))
    solution_sizes = []

    for prediction in predictions.transpose():
        potentialSolution = buildMDS(g, prediction)
        bestSolution = potentialSolution if len(potentialSolution) < len(bestSolution) else bestSolution
        solution_sizes.append(len(potentialSolution))
    
    return bestSolution, (time.time()-start), solution_sizes

def buildMDS(g, prediction):
    sortedNodes = sorted(enumerate(prediction), key=lambda x: x[1], reverse=True)
    nodeOrder = [x[0] for x in sortedNodes]

    # Build minimum dominating set using binary search
    min = 0
    max = len(nodeOrder) - 1
    while min < max:
        mid = (max + min) // 2
        currSolution = nodeOrder[:mid+1]

        if nx.algorithms.dominating.is_dominating_set(g, currSolution):
            max = mid
        else:
            min = mid + 1

    currSolution = nodeOrder[:min+1]
    assert(min == max)
    assert(nx.algorithms.dominating.is_dominating_set(g,currSolution))

    # Prune the dominating set
    while True:
        remove = False

        for i in range(len(currSolution) - 1, -1, -1):
            newSolution = currSolution[:i] + currSolution[i+1:]

            if nx.algorithms.dominating.is_dominating_set(g, newSolution):
                remove = True
                currSolution = newSolution
                break

        if not remove:
            break
    
    return sorted(currSolution)

# tuples stored as (gamma, greedy, GCN)
testing_analysis = {}

for graph_size in np.arange(500, 1001, 10):
    for edge_prob in [0.01, 0.015, 0.02]:

        graph = nx.erdos_renyi_graph(graph_size, edge_prob)
        adj = nx.adjacency_matrix(graph).todense()

        print(f"Generating greedy solution for graph of {graph_size} nodes and density {edge_prob}")
        greedyStart = time.time()
        tmpGreedy = list(nx.algorithms.dominating.dominating_set(graph))
        greedyTime = time.time() - greedyStart
        print("Found greedy solution")

        print(f"Getting GCN solutions")
        nn = adj.shape[0]

        startTime = time.time()
        features = np.ones([nn, N_bd])
        features = sp.lil_matrix(features)
        features = preprocess_features(features)
        support = simple_polynomials(adj, FLAGS.max_degree)

        tmpRuntime, outs = testingEvaluataion(features, support, placeholders)

        sol, totalTime, solution_sizes = getBestMDS(adj, outs)
        print(f"Found GCN solutions")

        testing_analysis[f"{graph_size}_{edge_prob}"] = {
            'best': len(sol),
            'median': int(median(solution_sizes)),
            'greedy': len(tmpGreedy),
            'runtime': time.time() - startTime,
            'greedy_time': greedyTime,
            'all': solution_sizes
        }

        with open(f'final-extend-results.json', "w") as f:
            json.dump(testing_analysis, f, indent=2)