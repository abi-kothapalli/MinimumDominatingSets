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

    for prediction in predictions.transpose():
        potentialSolution = buildMDS(g, prediction, set())
        bestSolution = potentialSolution if len(potentialSolution) < len(bestSolution) else bestSolution
    
    return bestSolution

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