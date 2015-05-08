import random
import numpy as np
import cv2
def featuredisplay(Image11,Image22,matched,features,nextFeatures):
    
    Image1 = np.copy(Image11)
    Image2 = np.copy(Image22)
    for j in range(len(matched)):
        R = j * random.randint(0, 65536) % 255
        G = j * random.randint(0, 65536) % 255
        B = j * random.randint(0, 65536) % 255
        startPt = tuple(np.array(features[matched[j][0]][0:2], dtype=np.int64))
        nextStartPt = tuple(np.array(nextFeatures[matched[j][1]][0:2], dtype=np.int64))
        cv2.circle(Image1, startPt, 10, (R, G, B), 2)
        cv2.circle(Image2, nextStartPt, 10, (R, G, B), 2)
        lineLength = 15
        endPt = (np.round(lineLength * np.cos(features[matched[j][0]][2]) + startPt[0]), np.round(lineLength * np.sin(features[matched[j][0]][2]) + startPt[1]))
        endPt = tuple(np.array(endPt, dtype=np.int64))
        cv2.line(Image1, startPt, endPt, (R, G, B), 2)
        nextEndPt = (np.round(lineLength * np.cos(nextFeatures[matched[j][1]][2]) + nextStartPt[0]), np.round(lineLength * np.sin(nextFeatures[matched[j][1]][2]) + nextStartPt[1]))
        nextEndPt = tuple(np.array(nextEndPt, dtype=np.int64))
        cv2.line(Image2, nextStartPt, nextEndPt, (R, G, B), 2)
    cv2.namedWindow('Test1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Test2', cv2.WINDOW_NORMAL)
    cv2.imshow('Test1', Image1)
    cv2.imshow('Test2', Image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
	
