import numpy as np
from scipy import ndimage
from scipy import misc

ORIGINAL_IMAGE_PATH = 'original_images/'

TOTAL_IMAGES = 24

images = []
for i in range(TOTAL_IMAGES):
    imagePath = ORIGINAL_IMAGE_PATH + str(i + 1) + '.JPG'
    print imagePath
    image = np.rot90(misc.imread(imagePath, True))
    gaussImage = ndimage.filters.gaussian_filter(image, 10)
    gradY, gradX = np.gradient(gaussImage)
    misc.imshow(gradX)
    misc.imshow(gradY)
