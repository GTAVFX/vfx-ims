import numpy as np
import cv2

def linearBlending(img1BlendArea, img2BlendArea):
    areaHeight, areaWidth, colorChannels = img1BlendArea.shape

    # Calculate Weight map for each image
    img1BlendWeightInterval = np.array(np.linspace(1, 0, areaWidth))
    img2BlendWeightInterval = img1BlendWeightInterval[::-1]

    img1BlendChannelWeight = np.tile(img1BlendWeightInterval.reshape(1, areaWidth), (areaHeight, 1))
    img2BlendChannelWeight = np.tile(img2BlendWeightInterval.reshape(1, areaWidth), (areaHeight, 1))
    img1BlendWeight = np.dstack((img1BlendChannelWeight, img1BlendChannelWeight, img1BlendChannelWeight))
    img2BlendWeight = np.dstack((img2BlendChannelWeight, img2BlendChannelWeight, img2BlendChannelWeight))

    # Make masks for both non-zero areas and at-least-one-zero areas
    img1NonZero = ((img1BlendArea[:, :, 0] != 0) * (img1BlendArea[:, :, 1] != 0) * (img1BlendArea[:, :, 2] != 0))
    img2NonZero = ((img2BlendArea[:, :, 0] != 0) * (img2BlendArea[:, :, 1] != 0) * (img2BlendArea[:, :, 2] != 0))
    imgNonZero = img1NonZero * img2NonZero
    kernel = np.ones([7, 7], dtype=np.uint8)
    imgNonZero = ~np.array(cv2.morphologyEx(np.array(~imgNonZero, dtype=np.uint8), cv2.MORPH_OPEN, kernel), dtype=np.bool)

    nonZeroMask = np.dstack((imgNonZero, imgNonZero, imgNonZero))
    atLeastOneZeroMask = ~nonZeroMask

    # Combining the results
    atLeastOneZeroArea = img1BlendArea * atLeastOneZeroMask + img2BlendArea * atLeastOneZeroMask
    nonZeroArea = (img1BlendArea * img1BlendWeight + img2BlendArea * img2BlendWeight) * nonZeroMask
    blendedArea = atLeastOneZeroArea + nonZeroArea
    return blendedArea