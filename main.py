import numpy as np
import cv2
from scipy import ndimage
from scipy import misc
from scipy import stats

ORIGINAL_IMAGE_PATH = 'original_images/'

TOTAL_IMAGES = 24

images = []
for i in range(TOTAL_IMAGES):
    imagePath = ORIGINAL_IMAGE_PATH + str(i + 1) + '.JPG'
    print imagePath
    image = np.rot90(misc.imread(imagePath, True))
    cvImage = cv2.imread(imagePath)
    cvImage = cv2.flip(cvImage, 1)
    cvImage = cv2.transpose(cvImage)

    imageHeight, imageWidth = image.shape
    gaussImage = ndimage.filters.gaussian_filter(image, 10)
    gradY, gradX = np.gradient(gaussImage)
    gradXX = gradX * gradX
    gradYY = gradY * gradY
    gradXY = gradX * gradY
    gauXX = ndimage.filters.gaussian_filter(gradXX, 5)
    gauYY = ndimage.filters.gaussian_filter(gradYY, 5)
    gauXY = ndimage.filters.gaussian_filter(gradXY, 5)

    k = 0.04
    Det = gauXX * gauYY - gauXY * gauXY
    Trace = gauXX + gauYY
    R = Det - k * Trace * Trace

    ThresR = stats.threshold(R, 0.5)
    ThresRdilate = ndimage.grey_dilation(ThresR, footprint=[[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    ThresRpeak = ThresR > ThresRdilate

    features = [(j, i) for i in range(imageHeight) for j in range(imageWidth) if (ThresRpeak[i][j])]
    for feature in features:
        cv2.circle(cvImage, feature, 10, (255, 0, 0), -1)

    misc.imshow(cvImage)
