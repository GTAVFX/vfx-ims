import numpy as np
from scipy import ndimage
from scipy import stats

# Parameters for feature detection
k = 0.04
thresholdR = 0.5

def detectFeatures(grayImage):
    print 'Detecting features for image...'
    gaussImage = ndimage.filters.gaussian_filter(grayImage, 5)
    gradY, gradX = np.gradient(gaussImage)
    gradXX = gradX * gradX
    gradYY = gradY * gradY
    gradXY = gradX * gradY
    gauXX = ndimage.filters.gaussian_filter(gradXX, 5)
    gauYY = ndimage.filters.gaussian_filter(gradYY, 5)
    gauXY = ndimage.filters.gaussian_filter(gradXY, 5)

    Det = gauXX * gauYY - gauXY * gauXY
    Trace = gauXX + gauYY
    R = Det - k * Trace * Trace

    ThresR = stats.threshold(R, thresholdR)
    ThresRdilate = ndimage.grey_dilation(ThresR, footprint=[[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    ThresRpeak = ThresR > ThresRdilate

    features = ThresRpeak.nonzero()
    features = zip(features[1], features[0]) #(column, row), (width, height) (x, y)
    print 'Detected', len(features), 'features'
    return features