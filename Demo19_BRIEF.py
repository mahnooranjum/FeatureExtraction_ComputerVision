##==============================================================================
##   Demo By: Mahnoor Anjum
##   Date: 31/03/2019
##   Codes inspired by:
##   Github.com/imvinod/
##   Official Documentation
##==============================================================================

import cv2
import numpy as np


#BRIEF
img = cv2.imread('imgs/demo19.jpg')
img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

cv2.imshow('BRIEF',img)
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fast = cv2.FastFeatureDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
keypoints=fast.detect(gray,None)
keypoints, descriptors = brief.compute(gray,keypoints)
cv2.drawKeypoints(gray, keypoints, img , flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('BRIEF', img)
print("BRIEF Keypoints ", len(keypoints))


cv2.waitKey()
cv2.destroyAllWindows()

