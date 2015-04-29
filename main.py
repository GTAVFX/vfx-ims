import numpy as np
import cv2
import featureDetect as fd
import featureDescriptor as fdes
import featureMatch as fm
import random

ORIGINAL_IMAGE_PATH = 'original_images/'

TOTAL_IMAGES = 24

imageFeatures = []
imageFeaturePatches = []
cv2.startWindowThread()
cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
for i in range(TOTAL_IMAGES):
    if i is 2:
        break
    imagePath = ORIGINAL_IMAGE_PATH + str(i + 1) + '.JPG'
    print imagePath
    colorImage = cv2.imread(imagePath)
    colorImage = cv2.flip(colorImage, 1)
    colorImage = cv2.transpose(colorImage)
    grayImage = cv2.cvtColor(np.float32(colorImage), cv2.COLOR_BGR2GRAY)
    features = fd.detectFeatures(grayImage)
    featurePatch = fdes.getFeaturePatch(grayImage, features)
    imageFeatures.append(features)
    imageFeaturePatches.append(featurePatch)
    for feature in features:
        cv2.circle(colorImage, feature[::-1], 10, (255, 0, 0), -1)
    cv2.imshow('Test', colorImage)
    cv2.waitKey(0)
matchedIndices = fm.featureMatch(imageFeaturePatches)

colorImage0 = cv2.imread('original_images/1.JPG')
colorImage0 = cv2.flip(colorImage0, 1)
colorImage0 = cv2.transpose(colorImage0)
colorImage1 = cv2.imread('original_images/2.JPG')
colorImage1 = cv2.flip(colorImage1, 1)
colorImage1 = cv2.transpose(colorImage1)
images = [colorImage0, colorImage1]

for i in range(1):
    nextIndex = (i + 1) if i is not len(matchedIndices) - 1 else 0
    matched = matchedIndices[i]
    features = imageFeatures[i]
    nextFeatures = imageFeatures[nextIndex]
    for j in range(len(matched)):
        R = j * random.randint(0, 65536) % 255
        G = j * random.randint(0, 65536) % 255
        B = j * random.randint(0, 65536) % 255
        cv2.circle(images[i], features[j][::-1], 20, (R, G, B), -1)
        cv2.circle(images[nextIndex], nextFeatures[matched[j]][::-1], 20, (R, G, B), -1)

cv2.namedWindow('Test1', cv2.WINDOW_NORMAL)
cv2.namedWindow('Test2', cv2.WINDOW_NORMAL)
cv2.imshow('Test1', images[0])
cv2.imshow('Test2', images[1])
cv2.waitKey(0)

cv2.destroyAllWindows()