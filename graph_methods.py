import time
import random
import networkx as nx
from statistics import mean

def pruneSet(currSolution, g):
    for i in range(len(currSolution) - 1, -1, -1):
        # newSolution = currSolution[:i] + currSolution[i+1:]
        currSolution[i], currSolution[-1] = currSolution[-1], currSolution[i]
        tmp = currSolution.pop()

        if not nx.algorithms.dominating.is_dominating_set(g, currSolution):
            currSolution.append(tmp)
                
    assert(nx.algorithms.dominating.is_dominating_set(g, currSolution))

def getBestMDS(adj, predictions):
    start = time.time()
    g = nx.from_numpy_matrix(adj)

    bestSolution = list(g)
    solution_sizes = []

    runtimes = []

    for prediction in predictions.transpose():
        tmpTime = time.time()
        potentialSolution = buildMDS(g, prediction)
        bestSolution = potentialSolution if len(potentialSolution) < len(bestSolution) else bestSolution
        solution_sizes.append(len(potentialSolution))
        runtimes.append(time.time() - tmpTime)
    
    return bestSolution, (time.time()-start), solution_sizes, mean(runtimes)

def buildMDS(g, prediction):
    sortedNodes = sorted(zip(list(g), prediction), key=lambda x: x[1], reverse=True)
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

    pruneSet(currSolution, g)
    
    return sorted(currSolution)

def greedySolution(adj):
    start_time = time.time()

    g = nx.from_numpy_matrix(adj)

    dominatingSet = []
    dominatedNodes = set()

    while not nx.algorithms.dominating.is_dominating_set(g, dominatingSet):
        currNodes = []
        max_white_nodes = 0
        for v in g:
            if v in dominatingSet:
                continue

            white_count = 0
            for u in g.neighbors(v):
                if u not in dominatedNodes:
                    white_count += 1
            if v not in dominatedNodes:
                white_count += 1
            
            if white_count == max_white_nodes:
                currNodes.append(v)
            elif white_count > max_white_nodes:
                currNodes = [v]
                max_white_nodes = white_count
        
        dominatingSet.extend(currNodes)
        for node in currNodes:
            dominatedNodes.add(node)
            for adj_node in g.neighbors(node):
                dominatedNodes.add(adj_node)
    
    greedySize = len(dominatingSet)

    # Prune the dominating set
    pruneSet(dominatingSet, g)
    
    assert(nx.algorithms.dominating.is_dominating_set(g, dominatingSet))

    prunedGreedySize = len(dominatingSet)

    return greedySize, prunedGreedySize, time.time() - start_time


def getPartialGreedy(adj, predictions, percent_greedy = 0.5):
    start = time.time()
    g = nx.from_numpy_matrix(adj)

    numNodes = adj.shape[0]

    dominatingSet = []
    dominatedNodes = set()

    while len(dominatedNodes) < numNodes * percent_greedy:
        currNodes = []
        max_white_nodes = 0
        for v in g:
            if v in dominatedNodes:
                continue

            white_count = 0
            for u in g.neighbors(v):
                if u not in dominatedNodes:
                    white_count += 1
            if v not in dominatedNodes:
                white_count += 1
            
            if white_count == max_white_nodes:
                currNodes.append(v)
            elif white_count > max_white_nodes:
                currNodes = [v]
                max_white_nodes = white_count
        
        dominatingSet.extend(currNodes)
        for node in currNodes:
            dominatedNodes.add(node)
            for adj_node in g.neighbors(node):
                dominatedNodes.add(adj_node)
    
    bestSolution = list(g)
    solution_sizes = []

    for prediction in predictions.transpose():
        potentialSolution = buildComboMDS(g, dominatingSet, prediction)
        bestSolution = potentialSolution if len(potentialSolution) < len(bestSolution) else bestSolution
        solution_sizes.append(len(potentialSolution))
    
    return bestSolution, (time.time()-start), solution_sizes    

def buildRandomSolution(adj):
    g = nx.from_numpy_matrix(adj)
    randomNodes = list(g)
    random.shuffle(randomNodes)

    min = 0
    max = len(randomNodes) - 1
    while min < max:
        mid = (max + min) // 2
        currSolution = randomNodes[:mid+1]

        if nx.algorithms.dominating.is_dominating_set(g, currSolution):
            max = mid
        else:
            min = mid + 1
    
    currSolution = randomNodes[:min+1]
    assert(min == max)
    assert(nx.algorithms.dominating.is_dominating_set(g,currSolution))

    pruneSet(currSolution, g)
    
    return sorted(currSolution)


def getCombos(adj, predictions, percent_random = 0.5):
    start = time.time()
    g = nx.from_numpy_matrix(adj)

    numNodes = adj.shape[0]
    randomNodes = list(g)
    random.shuffle(randomNodes)

    currNodes = []
    currNeighbors = set()

    idx = 0
    while idx < len(randomNodes) and len(currNeighbors) < numNodes * percent_random:
        currNodes.append(randomNodes[idx])
        currNeighbors.add(randomNodes[idx])
        for v in g.neighbors(randomNodes[idx]):
            currNeighbors.add(v)
        
        idx += 1
    
    bestSolution = list(g)
    solution_sizes = []

    for prediction in predictions.transpose():
        potentialSolution = buildComboMDS(g, currNodes, prediction)
        bestSolution = potentialSolution if len(potentialSolution) < len(bestSolution) else bestSolution
        solution_sizes.append(len(potentialSolution))
    
    return bestSolution, (time.time()-start), solution_sizes

def buildComboMDS(g, currNodes, prediction):
    sortedNodes = sorted(enumerate(prediction), key=lambda x: x[1], reverse=True)
    currNodeSet = set(currNodes)
    nodeOrder = []
    for x in sortedNodes:
        if x[0] not in currNodeSet:
            nodeOrder.append(x[0])

    # Build minimum dominating set using binary search
    min = 0
    max = len(nodeOrder) - 1
    while min < max:
        mid = (max + min) // 2
        currSolution = nodeOrder[:mid+1] + currNodes

        if nx.algorithms.dominating.is_dominating_set(g, currSolution):
            max = mid
        else:
            min = mid + 1

    currSolution = nodeOrder[:min+1] + currNodes
    # assert(min == max, f"min is {min} and max is {max}")
    assert(nx.algorithms.dominating.is_dominating_set(g,currSolution))

    # Prune the dominating set
    pruneSet(currSolution, g)
    
    return sorted(currSolution)