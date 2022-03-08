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

def generateExactSolutions(graph, num_solutions = 3, mute=False):
    prob = pulp.LpProblem("Minimum-Dominating-Set", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("y", graph.nodes(), cat=pulp.LpBinary)

    prob += pulp.lpSum(y)
    for v in graph.nodes():
        prob += (y[v] + pulp.lpSum([y[u] for u in graph.neighbors(v)]) >= 1)
        
    solutions = []

    for i in range(num_solutions):

        print(f"Working on solution number {i + 1}")
        start = timeit.default_timer()

        prob.solve(pulp.apis.COIN(msg=not mute))
        
        selected_vars = []
        for var in y:
             if y[var].value() != 0:
                selected_vars.append(var)
        
        solutions.append(selected_vars)

        print(f"Found solution number {i + 1} in {timeit.default_timer() - start} seconds")
        
        # Add a new constraint to prevent same solution from being found again
        prob += (pulp.lpSum([y[var] for var in selected_vars]) <= len(selected_vars) - 1)
        
        if pulp.LpStatus[prob.status] != 'Optimal':
            break
    
    return solutions
            
def generate_dataset(args):

    id = args.id

    graph_sizes = graphSizes(args.nodes, args.maxNodes, args.nodeStep)
    average_degrees = averageDegrees(args.degree, args.maxDegree, args.degreeStep)
        
    for nodes in graph_sizes:
        for avg_degree in average_degrees:

            print(f"Generating graph {id} with {nodes} nodes and {avg_degree} average degree")
            
            start_time = timeit.default_timer()

            edge_prob = getEdgeProb(nodes, avg_degree)
            graph = generateGraph(nodes, edge_prob)

            greedy_solution = generateGreedySolution(graph)

            exact_start_time = timeit.default_timer()

            exact_solutions = generateExactSolutions(graph, mute=args.mute)

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

    parser.add_argument("-i", "--id", type=int, default=0, help="ID of the graph")
    parser.add_argument("-n", "--nodes", type=int, default=150, help="Starting number of nodes in the graph")
    parser.add_argument("-d", "--degree", type=int, default=5, help="Starting average degree of the graph")

    parser.add_argument("-mn", "--maxNodes", type=int, default=500, help="Maximum number of nodes in the graph")
    parser.add_argument("-md", "--maxDegree", type=int, default=15, help="Maximum average degree of the graph")

    parser.add_argument("-ns", "--nodeStep", type=int, default=5, help="Step size for the number of nodes")
    parser.add_argument("-ds", "--degreeStep", type=int, default=1, help="Step size for the average degree")

    parser.add_argument("-m", "--mute", action='store_true')
    parser.set_defaults(mute=False)


    args = parser.parse_args()

    for PATH in OUTPUT_PATHS:
        if not os.path.isdir(PATH):
            os.mkdir(os.path.join(os.getcwd(), PATH))

    return args

if __name__ == "__main__":
    args = setup()
    generate_dataset(args)