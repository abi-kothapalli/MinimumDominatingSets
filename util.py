import networkx as nx
import numpy as np
from glob import glob


def read_adj_matrix(file):
    adj_mat = []

    with open(file, 'r') as f:
        n = int(f.readline())

        while True:
            line = f.readline()

            if line.isspace():
                continue
            
            if line.strip() == '-1':
                break

            adj_mat.append([int(x) for x in line.split()])
        
    return n, np.array(adj_mat)


def get_graphs_from_dir(dir):
    files = sorted(glob(f"{dir}/*.txt"))
    return {file.split("/")[-1]: {**read_adj_matrix(file), 'gamma': 'N/A', 'runtime': 'N/A'} for file in files}