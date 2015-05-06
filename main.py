import numpy as np
import cv2
import featureDetect as fd
import featureDescriptor as fdes
import featureMatch as fm
import orientationAssignment as oa
import cylintransform as ct
import util.calcImageDisplacements as cid
import imageStitching as ist
import random

ORIGINAL_IMAGE_PATH = 'original_images/'
# ORIGINAL_IMAGE_PATH = 'images/csie/'
OUTPUT_IMAGE_PATH = 'output_images/'
FOCUS = 2243

TOTAL_IMAGES = 9

images = []
imageFeatures = []
imageFeaturePatches = []
cv2.startWindowThread()
# cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
for i in range(TOTAL_IMAGES):
    imagePath = ORIGINAL_IMAGE_PATH + str(i + 1) + '.JPG'
    print imagePath
    colorImage = cv2.imread(imagePath)
    colorImage = ct.cylintransform(colorImage, FOCUS)
    images.append(colorImage)
    grayImage = cv2.cvtColor(np.float32(colorImage), cv2.COLOR_BGR2GRAY)
    featuresPos = fd.detectFeatures(grayImage)
    features = oa.assignOreintation(grayImage, featuresPos)
    featurePatch = fdes.getFeaturePatch(grayImage, features)
    imageFeatures.append(features)
    imageFeaturePatches.append(featurePatch)
    # for feature in features:
    #     startPt = tuple(np.array(feature[0:2], dtype=np.int64))
    #     lineLength = 15
    #     endPt = (np.round(lineLength * np.cos(feature[2]) + startPt[0]), np.round(lineLength * np.sin(feature[2]) + startPt[1]))
    #     endPt = tuple(np.array(endPt, dtype=np.int64))
    #     cv2.circle(colorImage, startPt, 10, (255, 0, 0), 2)
    #     cv2.line(colorImage, startPt, endPt, (255, 0, 0), 2)

    # cv2.imshow('Test', colorImage)
    # cv2.waitKey(0)
matchedIndices = fm.featureMatch(imageFeatures, imageFeaturePatches)

imageDisplacements = cid.calcImageDisplacements(imageFeatures, matchedIndices)
print len(images)
stitchedImage = ist.imageStitching(images, imageDisplacements)
outputPath = OUTPUT_IMAGE_PATH + 'stitchedImage.jpg'
cv2.imwrite(outputPath, stitchedImage)

for i in range(TOTAL_IMAGES - 1):
    nextIndex = i + 1
    colorImage = images[i]
    nextColorImage = images[nextIndex]
    # imagePath = ORIGINAL_IMAGE_PATH + str(i + 1) + '.JPG'
    # colorImage = cv2.imread(imagePath)
    # colorImage = ct.cylintransform(colorImage)
    # nextImagePath = ORIGINAL_IMAGE_PATH + str(nextIndex + 1) + '.JPG'
    # nextColorImage = cv2.imread(nextImagePath)
    # nextColorImage = ct.cylintransform(nextColorImage)
    # images = [colorImage, nextColorImage]

    matched = matchedIndices[i]
    features = imageFeatures[i]
    nextFeatures = imageFeatures[nextIndex]
    for j in range(len(matched)):
        R = j * random.randint(0, 65536) % 255
        G = j * random.randint(0, 65536) % 255
        B = j * random.randint(0, 65536) % 255
        startPt = tuple(np.array(features[matched[j][0]][0:2], dtype=np.int64))
        nextStartPt = tuple(np.array(nextFeatures[matched[j][1]][0:2], dtype=np.int64))
        cv2.circle(images[i], startPt, 10, (R, G, B), 2)
        cv2.circle(images[nextIndex], nextStartPt, 10, (R, G, B), 2)
        lineLength = 15
        endPt = (np.round(lineLength * np.cos(features[matched[j][0]][2]) + startPt[0]), np.round(lineLength * np.sin(features[matched[j][0]][2]) + startPt[1]))
        endPt = tuple(np.array(endPt, dtype=np.int64))
        cv2.line(images[i], startPt, endPt, (R, G, B), 2)
        nextEndPt = (np.round(lineLength * np.cos(nextFeatures[matched[j][1]][2]) + nextStartPt[0]), np.round(lineLength * np.sin(nextFeatures[matched[j][1]][2]) + nextStartPt[1]))
        nextEndPt = tuple(np.array(nextEndPt, dtype=np.int64))
        cv2.line(images[nextIndex], nextStartPt, nextEndPt, (R, G, B), 2)
    cv2.namedWindow('Test1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Test2', cv2.WINDOW_NORMAL)
    cv2.imshow('Test1', images[i])
    cv2.imshow('Test2', images[nextIndex])
    cv2.waitKey(0)

cv2.destroyAllWindows()