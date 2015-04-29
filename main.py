import numpy as np
import cv2
import featureDetect as fd
import featureDescriptor as fdes

ORIGINAL_IMAGE_PATH = 'original_images/'

TOTAL_IMAGES = 24

imageFeatures = []
cv2.startWindowThread()
cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
for i in range(TOTAL_IMAGES):
    imagePath = ORIGINAL_IMAGE_PATH + str(i + 1) + '.JPG'
    print imagePath
    colorImage = cv2.imread(imagePath)
    colorImage = cv2.flip(colorImage, 1)
    colorImage = cv2.transpose(colorImage)
    grayImage = cv2.cvtColor(np.float32(colorImage), cv2.COLOR_BGR2GRAY)
    features = fd.detectFeatures(grayImage)
    featurePatch = fdes.getFeaturePatch(grayImage, features)
    imageFeatures.append(features)
    for feature in features:
        cv2.circle(colorImage, feature[::-1], 10, (255, 0, 0), -1)
    cv2.imshow('Test', colorImage)
    cv2.waitKey(0)
cv2.destroyAllWindows()