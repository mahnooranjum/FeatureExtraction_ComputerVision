##==============================================================================
##   Demo By: Mahnoor Anjum
##   Date: 31/03/2019
##   Codes inspired by:
##   Github.com/imvinod/
##   Official Documentation
##==============================================================================

import cv2
import numpy as np

#ORB
img = cv2.imread('imgs/demo18.jpg')
cv2.imshow('ORB',img)
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create()
keypoints=orb.detect(gray,None)
keypoints, descriptors = orb.compute(gray,keypoints)
cv2.drawKeypoints(gray, keypoints, img , flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('ORB', img)
print("ORBs Keypoints ", len(keypoints))

cv2.waitKey()
cv2.destroyAllWindows()

