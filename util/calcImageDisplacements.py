import numpy as np

def calcImageDisplacements(allFeatures, allMatchedIndices):
    imageNum = len(allFeatures)

    imageDisplacemnts = []
    for i in range(imageNum - 1):
        nextIndex = i + 1
        featureIndices, matchedFeatureIndices = zip(*allMatchedIndices[i])
        featureIndices = np.array(featureIndices)
        matchedFeatureIndices = np.array(matchedFeatureIndices)

        matchedNum = len(featureIndices)

        featurePos = np.array([[allFeatures[i][index][0], allFeatures[i][index][1]] for index in featureIndices], dtype=np.int64)
        matchedFeaturePos = np.array([[allFeatures[nextIndex][index][0], allFeatures[nextIndex][index][1]] for index in matchedFeatureIndices], dtype=np.int64)

        displacement = featurePos - matchedFeaturePos
        averageDisplacement = np.sum(displacement, axis=0) / matchedNum

        imageDisplacemnts.append(tuple(averageDisplacement))

    return imageDisplacemnts