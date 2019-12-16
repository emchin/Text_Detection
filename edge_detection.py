#!/usr/bin/env python

import cv2
import numpy as np
img = cv2.imread('wb3.jpg')
blur = cv2.GaussianBlur(img,(5,5),0)
 
filter = cv2.Canny(img,100,200)
 
cv2.imshow('Original', img)
cv2.imshow('Gaussain filter',filter)
 
cv2.waitKey(0)
