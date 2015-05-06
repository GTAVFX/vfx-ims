import numpy as np
from scipy import ndimage

NUM_ANGLES = 8
NUM_BINS = 4
GRID_SPACING = 4
NUM_SAMPLES = NUM_BINS * GRID_SPACING
TOTAL_GRIDS = NUM_BINS * NUM_BINS
TOTAL_SAMPLES = NUM_SAMPLES * NUM_SAMPLES
FEATURE_WINDOW = 2 * GRID_SPACING

angles, ANGLE_SPACING = np.linspace(-np.pi, np.pi, NUM_ANGLES, endpoint=False, retstep=True)

# Default grid samples
gridsInterval = np.linspace(-6, 6, NUM_BINS)
gridsX, gridsY = np.meshgrid(gridsInterval, gridsInterval)
featureGrids = np.concatenate((gridsX.reshape(1, TOTAL_GRIDS), gridsY.reshape(1, TOTAL_GRIDS)), axis=0)
samplesInterval = np.linspace(-(2 * GRID_SPACING - 0.5), 2 * GRID_SPACING - 0.5, NUM_SAMPLES)
samplesX, samplesY = np.meshgrid(samplesInterval, samplesInterval)
featureSamples = np.concatenate((samplesX.reshape(1, TOTAL_SAMPLES), samplesY.reshape(1, TOTAL_SAMPLES)), axis=0)

def getFeaturePatch(grayImage, features):
    print 'Extracting SIFT descriptors...'
    imageHeight, imageWidth = grayImage.shape
    numFeatures = len(features)

    # Smoothing the image
    gaussImage = ndimage.filters.gaussian_filter(grayImage, 5)

    # Calculate gradient magnitudes and directions
    gradY, gradX = np.gradient(gaussImage)
    magnitude = np.sqrt(gradX ** 2 + gradY ** 2)
    direction = np.arctan2(gradY, gradX)
    direction[(direction == np.pi).nonzero()] = -np.pi

    descriptors = np.zeros([numFeatures, TOTAL_GRIDS * NUM_ANGLES])

    for i in range(numFeatures):
        # Rotate grid coordinates
        x, y, orientation = features[i]
        M = np.array([[np.cos(orientation), -np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]])
        featureRotGrids = np.dot(M, featureGrids) + np.tile(np.array([[x], [y]]), (1, TOTAL_GRIDS))
        featureRotSamples = np.dot(M, featureSamples) + np.tile(np.array([[x], [y]]), (1, TOTAL_SAMPLES))

        # Sampling magnitude and directions
        featureRotSamplesRound = np.round(featureRotSamples)
        featureRotSamplesIdx = np.array(featureRotSamplesRound[1, ] * imageWidth + featureRotSamplesRound[0, ], dtype=np.int64)
        magSamples = magnitude.take(featureRotSamplesIdx).reshape(TOTAL_SAMPLES, 1)
        dirSamples = direction.take(featureRotSamplesIdx).reshape(TOTAL_SAMPLES, 1)

        # Calculate position weightings
        featureRotGridsX = featureRotGrids[0].reshape(1, TOTAL_GRIDS)
        featureRotGridsY = featureRotGrids[1].reshape(1, TOTAL_GRIDS)
        featureRotSamplesX = featureRotSamples[0].reshape(TOTAL_SAMPLES, 1)
        featureRotSamplesY = featureRotSamples[1].reshape(TOTAL_SAMPLES, 1)

        distRotX = np.abs(np.tile(featureRotSamplesX, (1, TOTAL_GRIDS)) - np.tile(featureRotGridsX, (TOTAL_SAMPLES, 1)))
        distRotY = np.abs(np.tile(featureRotSamplesY, (1, TOTAL_GRIDS)) - np.tile(featureRotGridsY, (TOTAL_SAMPLES, 1)))

        weightX = distRotX / GRID_SPACING
        weightX = (1 - weightX) * (weightX <= 1)
        weightY = distRotY / GRID_SPACING
        weightY = (1 - weightY) * (weightY <= 1)
        weightPos = weightX * weightY

        # Calculate orientation weightings
        distDir = np.mod(np.tile(dirSamples, (1, NUM_ANGLES)) - np.tile(angles, (TOTAL_SAMPLES, 1)) - orientation + np.pi, 2 * np.pi) - np.pi

        weightDir = np.abs(distDir) / ANGLE_SPACING
        weightDir = (1 - weightDir) * (weightDir <= 1)

        # Calculate gaussian weightings
        weightGaussian = np.exp(-((featureRotSamplesX - x) ** 2 + ((featureRotSamplesY - y) ** 2)) / (2 * FEATURE_WINDOW ** 2)) / (2 * np.pi * FEATURE_WINDOW ** 2)

        # Creating feature descriptor
        weightedMags = np.tile(magSamples, (1, NUM_ANGLES)) * np.tile(weightGaussian, (1, NUM_ANGLES)) * weightDir
        currentSIFT = np.zeros([NUM_ANGLES, TOTAL_GRIDS])
        for a in range(NUM_ANGLES):
            weightedMag = weightedMags[:, a].reshape(TOTAL_SAMPLES, 1)
            weightedMag = np.tile(weightedMag, (1, TOTAL_GRIDS))
            currentSIFT[a, :] = np.sum(weightedMag * weightPos, axis=0)
        descriptors[i, :] = np.reshape(currentSIFT, (1, TOTAL_GRIDS * NUM_ANGLES))

    # Normalize SIFT descriptor
    norm = np.sqrt(np.sum(descriptors ** 2, axis=1))
    norm = np.reshape(norm, (norm.shape[0], 1))

    normDescriptors = descriptors / np.tile(norm, (1, TOTAL_GRIDS * NUM_ANGLES))

    # Suppress large gradients
    normDescriptors[(normDescriptors > 0.2).nonzero()] = 0.2

    # Renormalize SIFT descriptor
    norm = np.sqrt(np.sum(normDescriptors ** 2, axis=1))
    norm = np.reshape(norm, (norm.shape[0], 1))
    normDescriptors = normDescriptors / np.tile(norm, (1, TOTAL_GRIDS * NUM_ANGLES))

    return normDescriptors