import numpy as np
import cv2
from scipy import misc
import featureDetect as fd

ORIGINAL_IMAGE_PATH = 'original_images/'

TOTAL_IMAGES = 24

imageFeatures = []
cv2.startWindowThread()
cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
for i in range(TOTAL_IMAGES):
    imagePath = ORIGINAL_IMAGE_PATH + str(i + 1) + '.JPG'
    print imagePath
    image = np.rot90(misc.imread(imagePath, True))
    cvImage = cv2.imread(imagePath)
    cvImage = cv2.flip(cvImage, 1)
    cvImage = cv2.transpose(cvImage)
    features = fd.detectFeatures(image)
    imageFeatures.append(features)
    for feature in features:
        cv2.circle(cvImage, feature, 10, (255, 0, 0), -1)
    cv2.imshow('Test', cvImage)
    cv2.waitKey(0)
cv2.destroyAllWindows()