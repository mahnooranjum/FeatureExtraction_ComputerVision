##==============================================================================
##   Demo By: Mahnoor Anjum
##   Date: 31/03/2019
##   Codes inspired by:
##   Github.com/imvinod/
##   Official Documentation
##==============================================================================

import cv2
import numpy as np

#SIFT
img = cv2.imread('imgs/demo17.jpg')
cv2.imshow('SIFT',img)
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
keypoints=sift.detect(gray,None)
cv2.drawKeypoints(gray, keypoints, img , flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SIFT', img)
print("SIFT Keypoints ", len(keypoints))


cv2.waitKey()
cv2.destroyAllWindows()

