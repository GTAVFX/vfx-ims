from scipy import spatial

def featureMatch(allFeaturePatches):
    imageNum = len(allFeaturePatches)
    matchedIndices = []
    for i in range(imageNum):
        nextIndex = (i + 1) if i is not imageNum - 1 else 0
        tree = spatial.KDTree(allFeaturePatches[nextIndex])
        match = tree.query(allFeaturePatches[i])
        matchedIndices.append(match[1])
    return matchedIndices