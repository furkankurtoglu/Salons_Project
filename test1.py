# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:57:00 2020

@author: Furkan
"""

# importing OpenCV(cv2) module 
import cv2 
  
# Save image in set directory 
# Read RGB image 
img = cv2.imread('woman.jpg')  
  
# Output img with window name as 'image' 
cv2.imshow('image', img)  
  