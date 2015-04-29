PATCH_SIZE = 25

def getFeaturePatch(grayImage, features):
    imageHeight, imageWidth = grayImage.shape
    featurePatches = []
    for feature in features:
        if (feature[0] - PATCH_SIZE >= 0) and (feature[0] + PATCH_SIZE < imageHeight) and (feature[1] - PATCH_SIZE >= 0) and (feature[1] + PATCH_SIZE < imageWidth):
            featurePatch = grayImage[feature[0] - PATCH_SIZE / 2: feature[0] + PATCH_SIZE / 2 + 1, feature[1] - PATCH_SIZE / 2: feature[1] + PATCH_SIZE / 2 + 1]
            featurePatch = featurePatch.reshape(PATCH_SIZE * PATCH_SIZE)
            featurePatches.append(featurePatch)
    return featurePatches