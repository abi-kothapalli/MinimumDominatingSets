from __future__ import division
from __future__ import print_function
from statistics import median

import os
import time
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from copy import deepcopy

import tensorflow as tf
from gcn.utils import *
from gcn.models import GCN_DEEP_DIVER
from graph_methods import *
from iterative_greedy import iterativeGreedy

import networkx as nx

import json

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

# Load data
data_path = "./data"
train_mat_names = os.listdir(data_path)

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

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.outputs_softmax], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]

def generate_yy(exact_solutions, shape):
    res = np.zeros(shape)
    
    for n, sol in enumerate(exact_solutions): 
        res[sol, n] = 1
    return res

# Init variables
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

ckpt=tf.train.get_checkpoint_state(f"{RUN_NAME}")
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

# cost_val = []

all_loss = np.zeros(2000, dtype=float)
all_acc = np.zeros(2000, dtype=float)

epoch_loss = []
epoch_acc = []

# Train model
for epoch in range(FLAGS.epochs):
    if os.path.isdir(f"{RUN_NAME}/%04d"%epoch):
        continue
    ct = 0
    os.makedirs(f"{RUN_NAME}/%04d" % epoch)
    for idd in range(2000):
        id =  np.random.randint(len(train_mat_names))
        ct = ct + 1
        t = time.time()
        # load data
        mat_contents = sio.loadmat(data_path+'/'+train_mat_names[id])
        nn = int(train_mat_names[id].split("_")[1][1:])
        adj = mat_contents['adj']
        yy = mat_contents['exact']
        print(train_mat_names[id])
        yy = generate_yy(yy, (nn, yy.shape[0]))

        nn, nr = yy.shape # number of nodes & results
        # y_train = yy[:,np.random.randint(0,nr)]
        # y_train = np.concatenate([1-np.expand_dims(y_train,axis=1), np.expand_dims(y_train,axis=1)],axis=1)

        # sample an intermediate graph
        yyr = yy[:,np.random.randint(0,nr)]
        yyr_num = np.sum(yyr)

        yyr_down_num = np.random.randint(0,yyr_num)
        if yyr_down_num > 0:
            yyr_down_prob = yyr * np.random.random_sample(yyr.shape)
            yyr_down_flag = (yyr_down_prob >= np.partition(yyr_down_prob,-yyr_down_num)[-yyr_down_num])
            tmp = np.sum(adj[yyr_down_flag, :], axis=0) > 0
            tmp = np.asarray(tmp).reshape(-1)
            yyr_down_flag[tmp] = 1
            adj_down = adj[yyr_down_flag==0,:]
            adj_down = adj_down[:,yyr_down_flag==0]
            yyr_down = yyr[yyr_down_flag==0]
            adj = adj_down
            nn = yyr_down.shape[0]
            yyr = yyr_down

        y_train = np.concatenate([1 - np.expand_dims(yyr, axis=1), np.expand_dims(yyr, axis=1)], axis=1)

        features = np.ones([nn, N_bd])
        features = sp.lil_matrix(features)
        features = preprocess_features(features)
        support = simple_polynomials(adj, FLAGS.max_degree)

        train_mask = np.ones([nn,1], dtype=bool)

        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs], feed_dict=feed_dict)
        all_loss[ct-1] = outs[1]
        all_acc[ct-1] = outs[2]

        # Print results
        print('%03d %04d' % (epoch + 1, ct), "train_loss=", "{:.5f}".format(np.mean(all_loss[np.where(all_loss)])),
              "train_acc=", "{:.5f}".format(np.mean(all_acc[np.where(all_acc)])), "time=", "{:.5f}".format(time.time() - t))

    epoch_loss.append(np.mean(all_loss[np.where(all_loss)]))
    epoch_acc.append(np.mean(all_acc[np.where(all_acc)]))


    target=open(f"{RUN_NAME}/%04d/score.txt"%epoch,'w')
    target.write("%f\n%f\n"%(np.mean(all_loss[np.where(all_loss)]),np.mean(all_acc[np.where(all_acc)])))
    target.close()

    saver.save(sess,f"{RUN_NAME}/model.ckpt")
    saver.save(sess,f"{RUN_NAME}/%04d/model.ckpt"%epoch)

    with open(f"{RUN_NAME}/training_data.json", 'w') as f:
        json.dump({'Loss': epoch_loss, 'Accuracy': epoch_acc}, f)

# Define model evaluation function
def testingEvaluataion(features, support, placeholders):
    feed_dict_val = construct_feed_dict4pred(features, support, placeholders)
    outs_val = sess.run([model.outputs_softmax], feed_dict=feed_dict_val)
    return outs_val[0]

testing_path = "./test"
test_mat_names = sorted(os.listdir(testing_path))

# tuples stored as (gamma, greedy, GCN)
testing_analysis = {}

for test_mat_name in test_mat_names:
    mat_contents = sio.loadmat(testing_path+'/'+test_mat_name)

    adj = mat_contents['adj']
    g = nx.from_numpy_matrix(adj)
    gamma = mat_contents['gamma'][0][0]


    greedySize, greedyTime = greedySolution(g)
    randomSize, randomTime = randomSolution(g)

    startTime = time.time()
    nn = adj.shape[0]
    features = np.ones([nn, N_bd])
    features = sp.lil_matrix(features)
    features = preprocess_features(features)
    support = simple_polynomials(adj, FLAGS.max_degree)

    outs = testingEvaluataion(features, support, placeholders)

    sol, solution_sizes, avgTime = getBestMDS(g, outs)
    runtime = time.time() - startTime

    IGSize, IGTime = iterativeGreedy(g)

    randCombo = {}
    randTimes = []
    greedyCombo = {}
    greedyTimes = []

    for percent_random in np.concatenate((np.arange(0.7, 0.8501, 0.05), np.arange(0.86, 1.001, 0.01))):
        randSol, randComboTime = buildRandomCombo(g, outs, percent_random)
        greedySol, greedyComboTime = buildGreedyCombo(g, outs, percent_random)

        randCombo[round(percent_random, 2)] = len(randSol)
        randTimes.append(randComboTime)

        greedyCombo[round(percent_random, 2)] = len(greedySol)
        greedyTimes.append(greedyComboTime)

    testing_analysis[test_mat_name] = {
        'gamma': int(gamma),
        'best_gcn': len(sol),
        'iterative_greedy': len(IGSize),
        'gcn_solutions': solution_sizes,
        'greedy': greedySize,
        'random': randomSize,
        'random_combos': randCombo,
        'greedy_combos': greedyCombo,
        'gcn_runtime_total': runtime,
        'gcn_runtime_per_prediction': avgTime,
        'iterative_greedy_time': IGTime,
        'greedy_time': greedyTime,
        'random_time': randomTime,
        'random_combo_times': randTimes,
        'greedy_combo_times': greedyTimes,
    }

    with open(f'test-results.json', "w") as f:
        json.dump(testing_analysis, f, indent=2)

    print(f"Finished {test_mat_name}")