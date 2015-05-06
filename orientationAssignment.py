import numpy as np
import util.gaussianFilter as gf
from scipy import ndimage
import cv2

WINDOW_SIZE = 25
WINDOW_RADIUS = WINDOW_SIZE / 2
NUM_BINS = 36
HIST_ORIENT, HIST_STEP = np.linspace(-np.pi, np.pi, NUM_BINS, endpoint=False, retstep=True)
# ZERO_PAD = 30

def assignOreintation(grayImage, featuresPos):
    print 'Assigning orientations to each feature...'
    imageHeight, imageWidth = grayImage.shape
    numFeatures = len(featuresPos)

    # Smoothing the image
    gaussImage = ndimage.filters.gaussian_filter(grayImage, 5)

    # Calculate gradient magnitudes and directions
    # gaussian needed ???
    gradY, gradX = np.gradient(gaussImage)
    magnitude = np.sqrt(gradX ** 2 + gradY ** 2)
    direction = np.arctan2(gradY, gradX)
    direction[(direction == np.pi).nonzero()] = -np.pi

    # Padding image with extended zeros
    # padImage = np.zeros([imageHeight + 2 * ZERO_PAD, imageWidth + 2 * ZERO_PAD])
    # padMagnitude = np.zeros([imageHeight + 2 * ZERO_PAD, imageWidth + 2 * ZERO_PAD])
    # padDirection = np.zeros([imageHeight + 2 * ZERO_PAD, imageWidth + 2 * ZERO_PAD])
    # padImage[ZERO_PAD:-ZERO_PAD, ZERO_PAD:-ZERO_PAD] = gaussImage
    # padMagnitude[ZERO_PAD:-ZERO_PAD, ZERO_PAD:-ZERO_PAD] = magnitude
    # padDirection[ZERO_PAD:-ZERO_PAD, ZERO_PAD:-ZERO_PAD] = direction

    # Gaussian weighting mask
    g = gf.gaussianFilter(3)
    hf_sz = np.floor(len(g) / 2)
    g = np.outer(g, g)

    newFeaturePos = np.empty([0, 2])
    newFeatureOrientation = np.empty([0, 1])

    for i in range(numFeatures):
        y, x = featuresPos[i]

        # # Add zero padding to x, y
        # y += ZERO_PAD
        # x += ZERO_PAD
        if y - hf_sz < 0 or y + hf_sz >= imageHeight or x - hf_sz < 0 or x + hf_sz >= imageWidth:
            continue
        weight = g * magnitude[y - hf_sz:y + hf_sz + 1, x - hf_sz:x + hf_sz + 1]
        # weight = g * padMagnitude[y - hf_sz:y + hf_sz + 1, x - hf_sz:x + hf_sz + 1]
        directionWindow = direction[y - hf_sz:y + hf_sz + 1, x - hf_sz:x + hf_sz + 1]
        # directionWindow = padDirection[y - hf_sz:y + hf_sz + 1, x - hf_sz:x + hf_sz + 1]
        orientHist = np.zeros([NUM_BINS, 1])
        for bin in range(NUM_BINS):
            diff = np.mod(directionWindow - HIST_ORIENT[bin] + np.pi, 2 * np.pi) - np.pi
            orientHist[bin] = orientHist[bin] + np.sum(weight * np.maximum(1 - np.abs(diff) / HIST_STEP, 0))

        # Find peaks in orientation histogram
        peaks = orientHist
        rotRight = np.roll(peaks, 1)
        rotLeft = np.roll(peaks, -1)
        peaks[(peaks < rotRight).nonzero()] = 0
        peaks[(peaks < rotLeft).nonzero()] = 0
        maxPeakValue = np.max(peaks)
        maxPeakIndex = np.argmax(peaks)

        peakValue = maxPeakValue
        while peakValue > 0.8 * maxPeakValue:
            A = np.empty([0, 3])
            b = np.empty([0, 1])
            for j in range(-1, 2):
                A = np.concatenate((A, [[(HIST_ORIENT[maxPeakIndex] + HIST_STEP * j) ** 2, (HIST_ORIENT[maxPeakIndex] + HIST_STEP * j), 1]]), axis=0)
                bin = np.mod(maxPeakIndex + j + NUM_BINS - 1, NUM_BINS)
                b = np.concatenate((b, [orientHist[bin]]), axis=0)
            c = np.dot(np.linalg.pinv(A), b)
            maxOrient = -c[1] / (2 * c[0])
            while maxOrient < -np.pi:
                maxOrient += 2 * np.pi
            while maxOrient >= np.pi:
                maxOrient -= 2 * np.pi
            newFeaturePos = np.concatenate((newFeaturePos, [[y, x]]), axis=0)
            # newFeaturePos = np.concatenate((newFeaturePos, [[y - ZERO_PAD, x - ZERO_PAD]]), axis=0)
            newFeatureOrientation = np.concatenate((newFeatureOrientation, [maxOrient]), axis=0)

            peaks[maxPeakIndex] = 0
            peakValue = np.max(peaks)
            maxPeakIndex = np.argmax(peaks)
    newFeatures = np.concatenate((newFeaturePos, newFeatureOrientation), axis=1)
    # print newFeatures
    return newFeatures