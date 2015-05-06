from math import *
import numpy as np
def cylintransform(colorImage):

    imageHeight, imageWidth, colorChannels = colorImage.shape
    f = 1800
    s = f

    cyImage = np.zeros([imageHeight, imageWidth, colorChannels], dtype=np.uint8)
    midx = imageWidth/2
    midy = imageHeight/2
    for i in range(imageWidth):
        theta = atan2(i-midx, f)
        #print theta
        x = int(s * theta)
        for j in range(imageHeight):#i = x j = y
            y = int(floor(s * ((j-midy)/sqrt((i-midx)*(i-midx)+f*f))))
            cyImage[midy + y][midx + x] = colorImage[j][i]
    cyImage = np.array(cyImage, dtype=np.uint8)
    #print cyImage
    return cyImage
