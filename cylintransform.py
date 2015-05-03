from math import *
import numpy as np
def cylintransform(colorImage):

    imageHeight, imageWidth, colorChannels = colorImage.shape
    f = 696
    s = f

    cyImage = np.zeros([imageHeight, imageHeight, colorChannels], dtype=np.uint8)


    for i in range(imageWidth):
        theta = atan2(i, f)
        x = int(s * theta)
        for j in range(imageHeight):#i = x j = y
            y = int(s * (j/sqrt(x*x+f*f)))
            cyImage[y][x] = colorImage[j][i]
    cyImage = np.array(cyImage, dtype=np.uint8)
    print cyImage
    return cyImage
