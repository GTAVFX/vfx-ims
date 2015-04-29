from scipy import spatial

thresHold = 600

def featureMatch(allFeaturePatches):
    imageNum = len(allFeaturePatches)
    matchedIndices = []
    for i in range(imageNum):
        nextIndex = (i + 1) if i != imageNum - 1 else 0
        thisTree = spatial.KDTree(allFeaturePatches[i])
        nextTree = spatial.KDTree(allFeaturePatches[nextIndex])
        thisMatch = nextTree.query(allFeaturePatches[i])
        nextMatch = thisTree.query(allFeaturePatches[nextIndex])
        match = [thisMatch[1][index] for index in range(len(thisMatch[1])) if index == nextMatch[1][thisMatch[1][index]] and thisMatch[0][index] < thresHold]
        matchedIndices.append(match)
    return matchedIndices