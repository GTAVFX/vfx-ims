import numpy as np
import cv2
import featureDetect as fd
import featureDescriptor as fdes
import featureMatch as fm
import orientationAssignment as oa
import cylintransform as ct
import util.calcImageDisplacements as cid
import imageStitching as ist
import util.featureDisplay as fdis
import sys
import os
import re

# Parsing command line arguments
# Usage:
#   python main.py [input original images path] [output stitched image path / or path with filename]
#   if [input original images path] [output stitched image path / or path with filename ] are not given,
#   default input/file path/filename are used
DEFAULT_ORIGINAL_IMAGE_PATH = 'original_images/'
DEFAULT_OUTPUT_IMAGE_PATH = 'output_images/'
DEFAULT_OUTPUT_IMAGE_FILENAME = 'stitchedImage.jpg'

CONFIG_FILENAME = 'focus.txt'

ORIGINAL_IMAGE_PATH = DEFAULT_ORIGINAL_IMAGE_PATH
OUTPUT_IMAGE_PATH = DEFAULT_OUTPUT_IMAGE_PATH
OUTPUT_IMAGE_FILENAME = OUTPUT_IMAGE_PATH + DEFAULT_OUTPUT_IMAGE_FILENAME  # We use this to output image file

INPUT_IMAGE_FILE_PATTEN = '(\w)*(\d)+(\.JPG$|\.jpg$)'
OUTPUT_IMAGE_FILE_PATTEN = '.+\.JPG$|.+.jpg$'

if len(sys.argv) > 2:
    if re.match(OUTPUT_IMAGE_FILE_PATTEN, sys.argv[2]):
        OUTPUT_IMAGE_FILENAME = sys.argv[2]
    else:
        OUTPUT_IMAGE_FILENAME = sys.argv[2] + DEFAULT_OUTPUT_IMAGE_FILENAME
if len(sys.argv) > 1:
    ORIGINAL_IMAGE_PATH = sys.argv[1]

CONFIG_PATH = ORIGINAL_IMAGE_PATH + CONFIG_FILENAME

fileList = [ORIGINAL_IMAGE_PATH + fileName for fileName in os.listdir(ORIGINAL_IMAGE_PATH) if re.match(INPUT_IMAGE_FILE_PATTEN, fileName)]

convert = lambda text: int(text) if text.isdigit() else text
alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

fileList.sort(key=alphanum_key)
print fileList

TOTAL_IMAGES = len(fileList)
print 'Total', TOTAL_IMAGES, 'images'


# Read the focus value
focusFile = open(CONFIG_PATH, 'r')
FOCUS = int(focusFile.readline())
print 'Focus:', FOCUS

images = []
imageFeatures = []
imageFeaturePatches = []
cv2.startWindowThread()
for imagePath in fileList:
    print '======', 'Processing', imagePath, '======'
    colorImage = cv2.imread(imagePath)
    colorImage = ct.cylintransform(colorImage, FOCUS)
    images.append(colorImage)
    grayImage = cv2.cvtColor(np.float32(colorImage), cv2.COLOR_BGR2GRAY)
    featuresPos = fd.detectFeatures(grayImage)
    features = oa.assignOreintation(grayImage, featuresPos)
    featurePatch = fdes.getFeaturePatch(grayImage, features)
    imageFeatures.append(features)
    imageFeaturePatches.append(featurePatch)

matchedIndices = fm.featureMatch(imageFeatures, imageFeaturePatches)

for i in range(TOTAL_IMAGES - 1):
    fdis.featureDisplay(images[i], images[i+1], matchedIndices[i], imageFeatures[i], imageFeatures[i+1])

imageDisplacements = cid.calcImageDisplacements(imageFeatures, matchedIndices)
stitchedImage = ist.imageStitching(images, imageDisplacements)

print 'Writing the result to', OUTPUT_IMAGE_FILENAME
cv2.imwrite(OUTPUT_IMAGE_FILENAME, stitchedImage)

cv2.destroyAllWindows()
