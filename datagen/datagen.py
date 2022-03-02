import numpy as np
import networkx as nx
import pulp
import timeit
import scipy.io as sio
import json
import os
import argparse

OUTPUT_PATHS = ["output/", "output/mat_files", "output/json_files"]

### Graph Helper Methods ###

def generateGraph(n, density, generator = nx.erdos_renyi_graph):
    g = generator(n, density)
    return g

def getAdjMatrix(graph):
    return nx.adjacency_matrix(graph).todense().tolist()

def getAvgDegree(graph):
    degrees = [item[1] for item in list(graph.degree())]
    return sum(degrees)/len(degrees)

def verifyMDS(graph, dominating_set):
    return nx.algorithms.dominating.is_dominating_set(graph, dominating_set)

def getEdgeProb(nodes, avg_degree):
    return avg_degree/(nodes - 1)

def averageDegrees(minimum = 5, maximum = 15, step = 1):
    return np.arange(minimum, maximum + 1, step)

# len(graphSizes(500, 1000, 10)) for large range
def graphSizes(minimum = 150, maximum = 500, step = 5):
    return np.arange(minimum, maximum + 1, step)


### Dominating Set Methods ###

def generateGreedySolution(graph):
    return list(nx.algorithms.dominating.dominating_set(graph))

def generateExactSolutions(graph, num_solutions = 3, show_messages=True):
    prob = pulp.LpProblem("Minimum-Dominating-Set", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("y", graph.nodes(), cat=pulp.LpBinary)

    prob += pulp.lpSum(y)
    for v in graph.nodes():
        prob += (y[v] + pulp.lpSum([y[u] for u in graph.neighbors(v)]) >= 1)
        
    solutions = []

    for i in range(num_solutions):
        prob.solve(pulp.apis.PULP_CBC_CMD(msg=show_messages))
        
        selected_vars = []
        for var in y:
             if y[var].value() != 0:
                selected_vars.append(var)
        
        solutions.append(selected_vars)
        
        # Add a new constraint to prevent same solution from being found again
        prob += (pulp.lpSum([y[var] for var in selected_vars]) <= len(selected_vars) - 1)
        
        if pulp.LpStatus[prob.status] != 'Optimal':
            break
    
    return solutions
            
def generate_dataset(args):

    if args.size == 'small':
        graph_sizes = graphSizes()
        average_degrees = averageDegrees()
        i = 1
    elif args.size == 'large':
        graph_sizes = graphSizes(500, 1000, 50)
        average_degrees = [5, 10, 15, 20]
        i = 10001

    for nodes in graph_sizes:
        for avg_degree in average_degrees:
            
            start_time = timeit.default_timer()

            edge_prob = getEdgeProb(nodes, avg_degree)
            graph = generateGraph(nodes, edge_prob)

            greedy_solution = generateGreedySolution(graph)

            exact_start_time = timeit.default_timer()

            exact_solutions = generateExactSolutions(graph, show_messages=False)

            end_time = timeit.default_timer()

            graph_data = {
                'n': int(nodes),
                'p': int(edge_prob),
                'avg_degree': getAvgDegree(graph),
                'gamma': len(exact_solutions[0]),
                'greedy_size': len(greedy_solution),
                'solution_runtime': end_time - exact_start_time,
                'total_runtime': end_time - start_time,
                'greedy': greedy_solution,
                'exact': exact_solutions,
                'adj': getAdjMatrix(graph),
            }

            file_name = f"{id:05}_n{nodes}_d{avg_degree}"

            sio.savemat(os.path.join(OUTPUT_PATHS[1], file_name), graph_data, appendmat=True)
            with open(os.path.join(OUTPUT_PATHS[2], f"{file_name}.json"), "w") as f:
                json.dump(graph_data, f)
            
            if id % 10 == 0:
                print(f"{id}/{len(graph_sizes)*len(average_degrees)}")

            id += 1

def setup():

    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--size", type=str, default=None, help="Size range of the graphs")

    args = parser.parse_args()

    for PATH in OUTPUT_PATHS:
        if not os.path.isdir(PATH):
            os.mkdir(os.path.join(os.getcwd(), PATH))

    if not args.size:
        args.size = 'small'

    if args.size not in ['small', 'large']:
        raise ValueError("Size must be either 'small' or 'large'")    

    return args

if __name__ == "__main__":
    args = setup()
    generate_dataset(args)