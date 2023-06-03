from graph_methods import getSupports, pruneSet, buildComboMDS, getBestMDS
import random
import time

def constructInitial(g):

    supports = getSupports(g)
    dominatingSet = list(supports)
    dominatedNodes = set()

    for node in dominatingSet:
        dominatedNodes.add(node)
        dominatedNodes.update(g.neighbors(node))

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
            dominatedNodes.update(g.neighbors(node))

    pruneSet(dominatingSet, g)

    return dominatingSet, set(supports)

def improvementUpdateStep(dominatingSet, supports, g):
    
    initSize = len(dominatingSet)
    random.shuffle(dominatingSet)

    def findReplacement(node, dominatingSet):
        solSet = set(dominatingSet)

        subset = None
        domination_num = 0
        for neigh in g.neighbors(node):
            if neigh in solSet:
                domination_num += 1
        # If no neighbors are dominators, must choose a neighbor
        if domination_num == 0:
            subset = set(g.neighbors(node))
        
        # Now find a replacement node
        for neigh in g.neighbors(node):
            domination_num = 0
            for neigh2 in g.neighbors(neigh):
                if neigh2 == node:
                    continue
                if neigh2 in solSet:
                    domination_num += 1
            if domination_num == 0:
                if subset is None:
                    subset = set(g.neighbors(neigh))
                else:
                    subset.intersection_update(set(g.neighbors(neigh)))

        if node in subset:
            subset.remove(node)
        
        if not subset:
            return None
        else:
            return subset.pop()

    for i in range(len(dominatingSet)):
        node = dominatingSet[i]
        if node in supports:
            continue

        replacement = findReplacement(node, dominatingSet)
        if replacement is None:
            continue

        dominatingSet[i] = replacement
        pruneSet(dominatingSet, g)
        if len(dominatingSet) < initSize:
            return True
    return False

def localImprovement(dominatingSet, supports, g):
    while improvementUpdateStep(dominatingSet, supports, g):
        pass
    return dominatingSet

def destruction(dominatingSet, supports, beta=0.2):

    numNodesToRemove = beta * (len(dominatingSet) - len(supports))
    removedNodes = 0

    random.shuffle(dominatingSet)
    idx = len(dominatingSet) - 1
    while removedNodes < numNodesToRemove:
        node = dominatingSet[idx]
        if node in supports:
            idx -= 1
            continue

        dominatingSet[idx], dominatingSet[-1] = dominatingSet[-1], dominatingSet[idx]
        dominatingSet.pop()
        removedNodes += 1
        idx -= 1
    
    return dominatingSet        

def constructSolution(dominatingSet, g):
    dominatedNodes = set()

    for node in dominatingSet:
        dominatedNodes.add(node)
        dominatedNodes.update(g.neighbors(node))

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
            dominatedNodes.update(g.neighbors(node))

    pruneSet(dominatingSet, g)
    return dominatingSet

def iterativeGreedy(g, max_iter=200, beta=0.2, time_limit=600):
    startTime = time.time()
    dominatingSet, supports = constructInitial(g)
    dominatingSet = localImprovement(dominatingSet, supports, g)

    bestSol = len(dominatingSet)
    numItersWithoutImprovement = 0

    while numItersWithoutImprovement < max_iter and time.time() - startTime < time_limit:
        dominatingSet = destruction(dominatingSet, supports, beta)
        dominatingSet = constructSolution(dominatingSet, g)
        dominatingSet = localImprovement(dominatingSet, supports, g)

        if len(dominatingSet) < bestSol:
            bestSol = len(dominatingSet)
            numItersWithoutImprovement = 0
        else:
            numItersWithoutImprovement += 1

    return dominatingSet, time.time() - startTime

def IG_GCN(g, predictions, max_iter=200, beta=0.2, time_limit=600):
    num_preds = len(predictions)
    idx = 1

    startTime = time.time()
    supports = getSupports(g)
    dominatingSet, _, _ = getBestMDS(g, predictions.transpose())
    bestSol = len(dominatingSet)

    dominatingSet = localImprovement(dominatingSet, supports, g)
    bestSol = min(bestSol, len(dominatingSet))

    numItersWithoutImprovement = 0

    while numItersWithoutImprovement < max_iter and time.time() - startTime < time_limit:
        dominatingSet = destruction(dominatingSet, supports, beta)
        dominatingSet = buildComboMDS(g, dominatingSet, predictions[idx])
        dominatingSet = localImprovement(dominatingSet, supports, g)

        idx = (idx + 1) % num_preds

        if len(dominatingSet) < bestSol:
            bestSol = len(dominatingSet)
            numItersWithoutImprovement = 0
        else:
            numItersWithoutImprovement += 1

    return bestSol, time.time() - startTime