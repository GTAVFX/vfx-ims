import numpy as np
import util.linearBlend as linearBlend

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

    # Putting image1 to the stitched image
    print 'Stitching Image 1'
    imageOrigin = np.array([0, -maxNegativeShift])
    stitchedImage[imageOrigin[1]: imageOrigin[1] + imageHeight, imageOrigin[0]: imageOrigin[0] + imageWidth, :] = images[0][:, :, :]

    for i in range(imageNum - 1):
        print 'Stitching Image', str(i + 2)
        imageOrigin += np.array(imageDisplacements[i])
        overlapWidth = imageWidth - imageDisplacements[i][0]
        overlapImg1 = stitchedImage[:, imageOrigin[0]: imageOrigin[0] + overlapWidth, :]
        overlapImg2 = np.zeros([stitchedImageHeight, overlapWidth, colorChannels], dtype=np.uint8)
        overlapImg2[imageOrigin[1]: imageOrigin[1] + imageHeight, :, :] = images[i + 1][:, 0: overlapWidth, :]
        blendedOverlap = linearBlend.linearBlending(overlapImg1, overlapImg2)
        stitchedImage[imageOrigin[1]: imageOrigin[1] + imageHeight, imageOrigin[0]: imageOrigin[0] + imageWidth, :] = images[i + 1][:, :, :]
        stitchedImage[:, imageOrigin[0]: imageOrigin[0] + overlapWidth, :] = blendedOverlap[:, :, :]
    return stitchedImage
