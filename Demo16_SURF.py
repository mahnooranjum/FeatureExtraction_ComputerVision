##==============================================================================
##   Demo By: Mahnoor Anjum
##   Date: 31/03/2019
##   Codes inspired by:
##   Github.com/imvinod/
##   Official Documentation
##==============================================================================

import cv2
import numpy as np

#SURF
img = cv2.imread('imgs/demo16.jpg')
cv2.imshow('SURF',img)
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hessianThreshold = 2000
surf = cv2.xfeatures2d.SURF_create(hessianThreshold)
keypoints, descriptors = surf.detectAndCompute(img,None)
cv2.drawKeypoints(gray, keypoints, img , flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SURF', img)
print("SURF Keypoints ", len(keypoints))


cv2.waitKey()
cv2.destroyAllWindows()

