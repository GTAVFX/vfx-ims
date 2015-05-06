import numpy as np

def imageStitching(images, imageDisplacements):
    imageNum = len(images)
    imageHeight, imageWidth, colorChannels = images[0].shape

    # Calculate width and height for stitched image
    stitchedImageWidth = imageWidth + np.sum(np.array(imageDisplacements)[:, 0])

    yOrigin = 0
    maxPositiveShift = 0
    maxNegativeShift = 0
    for i in range(imageNum - 1):
        yOrigin += imageDisplacements[i][1]
        if yOrigin > maxPositiveShift:
            maxPositiveShift = yOrigin
        if yOrigin < maxNegativeShift:
            maxNegativeShift = yOrigin
    stitchedImageHeight = imageHeight + (maxPositiveShift - maxNegativeShift)

    # Stitching images
    stitchedImage = np.zeros([stitchedImageHeight, stitchedImageWidth, colorChannels], dtype=np.uint8)
    imageOrigin = np.array([0, -maxNegativeShift])
    for i in range(imageNum):
        print 'Stitching Image', str(i + 1)
        for c in range(colorChannels):
            stitchedImage[imageOrigin[1]: imageOrigin[1] + imageHeight, imageOrigin[0]: imageOrigin[0] + imageWidth, c] = images[i][:, :, c]
        if i < imageNum - 1:
            imageOrigin += np.array(imageDisplacements[i])
    return stitchedImage
