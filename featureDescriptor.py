import numpy as np

PATCH_SIZE = 16
PATCH_RADIUS = PATCH_SIZE / 2
NUM_ANGLES = 8
NUM_BINS = 4
NUM_SAMPLES = NUM_BINS * NUM_BINS
ALPHA = 9

angles = np.linspace(0, 2 * np.pi, NUM_ANGLES, endpoint=False)

# Default grid samples
interval, GRID_RESOLUTION = np.linspace(2. / NUM_BINS, 2., NUM_BINS, retstep=True)
interval -= (1. / NUM_BINS + 1)
GRID_RESOLUTION *= PATCH_RADIUS
gridX, gridY = np.meshgrid(interval, interval)
gridX = np.reshape(gridX, (1, NUM_SAMPLES))
gridY = np.reshape(gridY, (1, NUM_SAMPLES))

def getFeaturePatch(grayImage, featuresPos):
    print 'Extracting SIFT descriptors...'
    imageHeight, imageWidth = grayImage.shape
    numFeatures = len(featuresPos)
    siftPatches = np.zeros([numFeatures, NUM_SAMPLES * NUM_ANGLES])

    # Calculate gradient magnitudes and directions
    # gaussian needed ???
    gradY, gradX = np.gradient(grayImage)
    magnitude = np.sqrt(gradX ** 2 + gradY ** 2)
    direction = np.arctan2(gradY, gradX)

    # Orientation images
    orientation = np.zeros([imageHeight, imageWidth, NUM_ANGLES])
    for a in range(NUM_ANGLES):
        tmp = np.cos(direction - angles[a]) ** ALPHA
        tmp *= (tmp > 0)
        orientation[:, :, a] = tmp * magnitude

    for i in range(len(featuresPos)):
        y, x = featuresPos[i]
        # Bin centers coordinates
        gridXT = gridX * PATCH_RADIUS + x
        gridYT = gridY * PATCH_RADIUS + y

        # Window of pixel of this descriptor
        # xLow = np.floor(np.maximum(x - PATCH_RADIUS - GRID_RESOLUTION/2, 0))
        # xHigh = np.ceil(np.minimum(x + PATCH_RADIUS + GRID_RESOLUTION/2, imageWidth - 1))
        # yLow = np.floor(np.maximum(y - PATCH_RADIUS - GRID_RESOLUTION/2, 0))
        # yHigh = np.ceil(np.minimum(y + PATCH_RADIUS + GRID_RESOLUTION/2, imageHeight - 1))
        xLow = np.maximum(x - PATCH_RADIUS, 0)
        xHigh = np.minimum(x + PATCH_RADIUS - 1, imageWidth - 1)
        yLow = np.maximum(y - PATCH_RADIUS, 0)
        yHigh = np.minimum(y + PATCH_RADIUS - 1, imageHeight - 1)

        windowWidth = xHigh - xLow + 1
        windowHeight = yHigh - yLow + 1
        numPixels = windowWidth * windowHeight

        # Pixel coordinates
        gridPX, gridPY = np.meshgrid(np.linspace(xLow, xHigh, windowWidth), np.linspace(yLow, yHigh, windowHeight))
        gridPX = np.reshape(gridPX, (numPixels, 1))
        gridPY = np.reshape(gridPY, (numPixels, 1))

        # Calculate distance between each pixel and grid sample
        distPX = np.abs(np.tile(gridPX, (1, NUM_SAMPLES)) - np.tile(gridXT, (numPixels, 1)))
        distPY = np.abs(np.tile(gridPY, (1, NUM_SAMPLES)) - np.tile(gridYT, (numPixels, 1)))

        # Calculate weight of contribution of each pixel to each bin
        weightX = distPX / (GRID_RESOLUTION / 2 + 1)
        weightX = (1 - weightX) * (weightX <= 1)
        weightY = distPY / (GRID_RESOLUTION / 2 + 1)
        weightY = (1 - weightY) * (weightY <= 1)
        weights = weightX * weightY

        # Make SIFT descriptor
        currentSIFT = np.zeros([NUM_ANGLES, NUM_SAMPLES])
        for a in range(NUM_ANGLES):
            tmp = np.reshape(orientation[yLow:yHigh+1, xLow:xHigh+1, a], (numPixels, 1))
            tmp = np.tile(tmp, (1, NUM_SAMPLES))
            currentSIFT[a, :] = np.sum(tmp * weights, axis=0)

        siftPatches[i, :] = np.reshape(currentSIFT, (1, NUM_SAMPLES * NUM_ANGLES))

    # Normalize SIFT descriptor
    norm = np.sqrt(np.sum(siftPatches ** 2, axis=1))
    # normIndices = (norm > 1).nonzero()
    norm = np.reshape(norm, (norm.shape[0], 1))
    # siftNormPatches = siftPatches[normIndices]

    siftNormPatches = siftPatches / np.tile(norm, (1, NUM_SAMPLES * NUM_ANGLES))
    # siftNormPatches = siftNormPatches / np.tile(norm[normIndices], (1, NUM_SAMPLES * NUM_ANGLES))

    # Suppress large gradients
    siftNormPatches[(siftNormPatches > 0.2).nonzero()] = 0.2

    # Renormalize SIFT descriptor
    norm = np.sqrt(np.sum(siftNormPatches ** 2, axis=1))
    norm = np.reshape(norm, (norm.shape[0], 1))
    siftNormPatches = siftNormPatches / np.tile(norm, (1, NUM_SAMPLES * NUM_ANGLES))

    # siftPatches[normIndices] = siftNormPatches
    # return siftPatches
    return siftNormPatches