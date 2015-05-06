import numpy as np
def cylintransform(colorImage, focus):

    imageHeight, imageWidth, colorChannels = colorImage.shape
    f = focus
    s = f

    midX = imageWidth / 2
    midY = imageHeight / 2

    coordX, coordY = np.meshgrid(np.arange(imageWidth), np.arange(imageHeight))
    shiftedCoordX = coordX - midX
    shiftedCoordY = coordY - midY
    theta = np.arctan2(shiftedCoordX, f)
    shiftedCyCoordX = np.array(s * theta, dtype=np.int64)
    shiftedCyCoordY = np.array(np.floor(s * (shiftedCoordY / np.sqrt(shiftedCoordX ** 2 + f ** 2))), dtype=np.int64)

    cyCoordX = shiftedCyCoordX + midX
    cyCoordY = shiftedCyCoordY + midY

    cyImage = np.zeros([imageHeight, imageWidth, colorChannels], dtype=np.uint8)

    imageIndices = (coordY * imageWidth + coordX).flatten()
    cyImageMappedIndices = (cyCoordY * imageWidth + cyCoordX).flatten()

    for c in range(colorChannels):
        cyImageChannel = np.zeros([imageHeight, imageWidth], dtype=np.uint8)
        cyImageChannel.put(cyImageMappedIndices, colorImage[:, :, c].take(imageIndices))
        cyImage[:, :, c] = cyImageChannel

    cyMinCoordX = np.min(cyCoordX)
    cyMaxCoordX = np.max(cyCoordX)
    cyMinCoordY = np.min(cyCoordY)
    cyMaxCoordY = np.max(cyCoordY)

    return cyImage[cyMinCoordY: cyMaxCoordY + 1, cyMinCoordX: cyMaxCoordX + 1]
