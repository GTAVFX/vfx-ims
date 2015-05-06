from scipy import spatial

THRESHOLD_1NN2NN_RATIO = 0.6

def featureMatch(allFeatures, allFeaturePatches):
    imageNum = len(allFeaturePatches)
    matchedIndices = []
    for i in range(imageNum):
        nextIndex = (i + 1) if i != imageNum - 1 else 0
        print 'Matching', str(len(allFeaturePatches[i])), 'features of image', str(i), 'and', str(len(allFeaturePatches[nextIndex])), 'features of image', str(nextIndex)
        thisTree = spatial.KDTree(allFeaturePatches[i])
        nextTree = spatial.KDTree(allFeaturePatches[nextIndex])
        thisMatch = nextTree.query(allFeaturePatches[i], k=2) # Query 1NN and 2NN
        nextMatch = thisTree.query(allFeaturePatches[nextIndex], k=2) # Query 1NN and 2NN

        thisRatio1NN2NN = thisMatch[0][:, 0] / thisMatch[0][:, 1]
        thisIndices = (thisRatio1NN2NN < THRESHOLD_1NN2NN_RATIO).nonzero()[0].flatten()
        thisMatchedIndices = thisMatch[1][thisIndices, 0].flatten()

        nextRatio1NN2NN = nextMatch[0][:, 0] / nextMatch[0][:, 1]
        nextIndices = (nextRatio1NN2NN < THRESHOLD_1NN2NN_RATIO).nonzero()[0].flatten()
        nextMatchedIndices = nextMatch[1][nextIndices, 0].flatten()

        thisMatchedPairs = zip(thisIndices, thisMatchedIndices)
        nextMatchedPairs = zip(nextMatchedIndices, nextIndices)

        thisMatchedSet = set(thisMatchedPairs)
        nextMatchedSet = set(nextMatchedPairs)
        match = [matchPair for matchPair in thisMatchedSet & nextMatchedSet]
        print 'Total', str(len(match)), 'pairs'

        matchedIndices.append(match)
    return matchedIndices