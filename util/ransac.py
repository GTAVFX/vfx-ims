import numpy as np
import random

RANSAC_LOOP_NUM = 100
SAMPLE_NUM = 4
DIFF_THRESHOLD = 20

def ransac(matchedIndexPairs, matchedFeaturePosPairs):
    pairNum = len(matchedIndexPairs)
    if pairNum <= SAMPLE_NUM:
        return matchedIndexPairs
    else:
        featurePos, matchedFeaturePos = zip(*matchedFeaturePosPairs)
        featurePos = np.array(featurePos)
        matchedFeaturePos = np.array(matchedFeaturePos)

        displacement = featurePos - matchedFeaturePos

        inliers = []
        inlierNums = []

        for k in range(RANSAC_LOOP_NUM):
            sampleIndices = random.sample(range(pairNum), SAMPLE_NUM)
            averageDisplacement = np.sum(displacement[sampleIndices, ], axis=0) / SAMPLE_NUM
            diffDisplacement = displacement - averageDisplacement
            diff = diffDisplacement[:, 0] ** 2 + diffDisplacement[:, 1] ** 2
            inlierIndices = (diff < DIFF_THRESHOLD).nonzero()[0].flatten()
            inliers.append(inlierIndices)
            inlierNums.append(len(inlierIndices))

        maxInliersIndex = np.argmax(inlierNums)
        maxInliers = inliers[maxInliersIndex]

        newMatchedIndexPairs = [matchedIndexPairs[index] for index in maxInliers]
        return newMatchedIndexPairs