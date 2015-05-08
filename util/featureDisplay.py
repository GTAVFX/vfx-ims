import random
import numpy as np
import cv2

def displayFeatures(image, imageFeatures):
    Image = np.array(image)
    for feature in imageFeatures:
        x, y, orientation = feature
        startPt = (int(x), int(y))
        cv2.circle(Image, startPt, 10, (255, 0, 0), 2)
        lineLength = 15
        endPt = (np.round(lineLength * np.cos(orientation) + startPt[0]), np.round(lineLength * np.sin(orientation) + startPt[1]))
        endPt = tuple(np.array(endPt, dtype=np.int64))
        cv2.line(Image, startPt, endPt, (255, 0, 0), 2)
    cv2.namedWindow('Features', cv2.WINDOW_NORMAL)
    cv2.imshow('Features', Image)
    cv2.waitKey(0)

def displayMatchedFeatures(image1, image2, matchedPairs, image1Features, image2Features):
    Image1 = np.array(image1)
    Image2 = np.array(image2)
    for j in range(len(matchedPairs)):
        R = j * random.randint(0, 65536) % 255
        G = j * random.randint(0, 65536) % 255
        B = j * random.randint(0, 65536) % 255
        startPt = tuple(np.array(image1Features[matchedPairs[j][0]][0:2], dtype=np.int64))
        nextStartPt = tuple(np.array(image2Features[matchedPairs[j][1]][0:2], dtype=np.int64))
        cv2.circle(Image1, startPt, 10, (R, G, B), 2)
        cv2.circle(Image2, nextStartPt, 10, (R, G, B), 2)
        lineLength = 15
        endPt = (np.round(lineLength * np.cos(image1Features[matchedPairs[j][0]][2]) + startPt[0]), np.round(lineLength * np.sin(image1Features[matchedPairs[j][0]][2]) + startPt[1]))
        endPt = tuple(np.array(endPt, dtype=np.int64))
        cv2.line(Image1, startPt, endPt, (R, G, B), 2)
        nextEndPt = (np.round(lineLength * np.cos(image2Features[matchedPairs[j][1]][2]) + nextStartPt[0]), np.round(lineLength * np.sin(image2Features[matchedPairs[j][1]][2]) + nextStartPt[1]))
        nextEndPt = tuple(np.array(nextEndPt, dtype=np.int64))
        cv2.line(Image2, nextStartPt, nextEndPt, (R, G, B), 2)
    cv2.namedWindow('Image 1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Image 2', cv2.WINDOW_NORMAL)
    cv2.imshow('Image 1', Image1)
    cv2.imshow('Image 2', Image2)
    cv2.waitKey(0)
