from __future__ import division
from __future__ import print_function

import sys
import random
import os
from collections import defaultdict
import time
from statistics import median
import json
from tracemalloc import start

import numpy as np
import tensorflow as tf
from gcn.utils import *
from gcn.models import GCN_DEEP_DIVER
from graph_methods import *

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


def get_real_graphs():
    GRAPH_PATH = "./REDDIT-MULTI-5K"

    def get_node_map():
        node_map = [None]

        with open(os.path.join(GRAPH_PATH, "REDDIT-MULTI-5K.graph_idx"), "r") as f:
            for node_id, graph_id in enumerate(f, 1):
                node_map.append(int(graph_id))

        return node_map

    def get_edge_lists(node_map = None):
        if not node_map:
            node_map = get_node_map()

        edge_lists = defaultdict(list)

        with open(os.path.join(GRAPH_PATH, "REDDIT-MULTI-5K.edges"), "r") as f:
            for line in f:
                n1, n2 = line.strip('\n').split(',')
                edge_lists[node_map[int(n1)]].append(f"{n1} {n2}")

        return edge_lists

    def create_graphs(edge_lists = None):

        if not edge_lists:
            edge_lists = get_edge_lists()

        graphs = {}

        for graph_id in edge_lists:
            graphs[graph_id] = nx.parse_edgelist(edge_lists[graph_id], nodetype=int)

        return graphs

    return create_graphs()

# Define model evaluation function
def testingEvaluataion(features, support, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict4pred(features, support, placeholders)
    outs_val = sess.run([model.outputs_softmax], feed_dict=feed_dict_val)
    return (time.time() - t_test), outs_val[0]

# tuples stored as (gamma, greedy, GCN)
testing_analysis = {}

# graphs = get_real_graphs()
# sorted_graphs = sorted(graphs.values(), key=lambda x: len(x))

PATH = "./../datasets/"
assert(sys.argv[1].endswith(".json"))
DATA_PATH = os.path.join(PATH, sys.argv[1])

with open(DATA_PATH) as f:
    data = json.load(f)

for graph_id in data:

    adj = np.array(data[graph_id]["adj"])

    print(f"Generating greedy solution for {graph_id}")
    updatedGreedy, prunedGreedy, newGreedyTime = greedySolution(adj)
    print("Found greedy solution")

    randomStart = time.time()
    randomSize = len(buildRandomSolution(adj))
    randomTime = time.time() - randomStart

    print(f"Getting GCN solutions")
    nn = adj.shape[0]

    startTime = time.time()
    features = np.ones([nn, N_bd])
    features = sp.lil_matrix(features)
    features = preprocess_features(features)
    support = simple_polynomials(adj, FLAGS.max_degree)

    tmpRuntime, outs = testingEvaluataion(features, support, placeholders)

    sol, totalTime, solution_sizes, avgTime = getBestMDS(adj, outs)
    runtime = time.time() - startTime
    print(f"Found GCN solutions")

    # combo = {}
    # medianCombo = {}
    # randTimes = []
    # for percent_random in [0.700, 0.800, 0.900, 0.95, 0.99, 1.000]:
    #     randSol, randTime, randSizes = getCombos(adj, outs, percent_random)
    #     combo[round(percent_random, 2)] = len(randSol)
    #     medianCombo[round(percent_random, 2)] = median(randSizes)
    #     randTimes.append(randTime)

    testing_analysis[graph_id] = {
        'size': data[graph_id]["n"],
        'gamma': data[graph_id]["gamma"],
        'best': len(sol),
        'pruned_greedy': prunedGreedy,
        'random': randomSize,
        # 'combo': combo,
        # 'medianCombo': medianCombo,
        'gamma_time': data[graph_id]["runtime"],
        'runtime': runtime,
        'total_prediction_time': totalTime,
        'runtime_per_prediction': avgTime,
        'greedy_time': newGreedyTime,   
        'random_time': randomTime,
        # 'combo_times': randTimes,  
        'all': solution_sizes,
    }

    with open(f'real-world-results-{sys.argv[1]}', "w") as f:
        json.dump(testing_analysis, f, indent=2)