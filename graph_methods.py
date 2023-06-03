import time
import random
import networkx as nx
from statistics import mean
from collections import defaultdict
import os

def getSupports(g):

    leaves = set()
    supports = set()

    for node in g:
        if node in supports or node in leaves:
            continue

        if g.degree(node) == 0:
            supports.add(node)
        elif g.degree(node) == 1:
            leaves.add(node)
            supports.add(next(g.neighbors(node)))
    
    return supports

def pruneSet(currSolution, g):

    assert nx.algorithms.dominating.is_dominating_set(g, currSolution), "Initial solution is not a dominating set"

    for i in range(len(currSolution) - 1, -1, -1):

        remNode = currSolution[i]
        isDominated =  False
        neighborsDominated = True
        for neigh in g.neighbors(remNode):
            if neigh in currSolution and neigh != remNode:
                # At least one neighbor must be in the solution if we are removing this node
                isDominated = True
            else:
                # If neighbor not in solution, one of the neighbor's neighbors (other than the node being removed) must be in the solution
                currDominated = False
                for neigh2 in g.neighbors(neigh):
                    if neigh2 == remNode:
                        continue
                    if neigh2 in currSolution:
                        currDominated = True
                        break
                if not currDominated:
                    neighborsDominated = False
                    break

        if isDominated and neighborsDominated:
            currSolution[i], currSolution[-1] = currSolution[-1], currSolution[i]
            currSolution.pop()

def getBestMDS(g, predictions):
    bestSolution = list(g)
    solution_sizes = []

    runtimes = []

    for prediction in predictions.transpose():
        tmpTime = time.time()
        potentialSolution = buildMDS(g, prediction, set())
        bestSolution = potentialSolution if len(potentialSolution) < len(bestSolution) else bestSolution
        solution_sizes.append(len(potentialSolution))
        runtimes.append(time.time() - tmpTime)
    
    return bestSolution, solution_sizes, mean(runtimes)

def buildMDS(g, prediction, supportNodes):
    sortedNodes = sorted(zip(list(g), prediction), key=lambda x: x[1], reverse=True)

    nodeOrder = list(supportNodes)
    for x in sortedNodes:
        if x not in supportNodes:
            nodeOrder.append(x[0])

    # Build minimum dominating set using binary search
    min = len(supportNodes)
    max = len(nodeOrder) - 1
    while min < max:
        mid = (max + min) // 2
        currSolution = nodeOrder[:mid+1]

        if nx.algorithms.dominating.is_dominating_set(g, currSolution):
            max = mid
        else:
            min = mid + 1

    currSolution = nodeOrder[:min+1]
    assert min == max
    assert nx.algorithms.dominating.is_dominating_set(g,currSolution)

    pruneSet(currSolution, g)
    
    return sorted(currSolution)

def greedySolution(g):
    start_time = time.time()

    dominatingSet = []
    dominatedNodes = set()

    while len(dominatedNodes) < len(g):
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

    # Prune the dominating set
    pruneSet(dominatingSet, g)
    
    assert nx.algorithms.dominating.is_dominating_set(g, dominatingSet)

    greedySize = len(dominatingSet)

    return greedySize, time.time() - start_time

def randomSolution(g):
    start_time = time.time()

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
    assert min == max
    assert nx.algorithms.dominating.is_dominating_set(g,currSolution)

    pruneSet(currSolution, g)
    
    return len(currSolution), time.time() - start_time

def buildGreedyCombo(g, predictions, percent_greedy = 0.5):
    start = time.time()

    numNodes = len(g)

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
    
    return bestSolution, (time.time()-start)

def buildRandomCombo(g, predictions, percent_random = 0.5):
    start = time.time()

    numNodes = len(g)
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
    
    return bestSolution, (time.time()-start)

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
    assert min == max, f"min is {min} and max is {max}"
    assert nx.algorithms.dominating.is_dominating_set(g,currSolution)

    # Prune the dominating set
    pruneSet(currSolution, g)
    
    return sorted(currSolution)