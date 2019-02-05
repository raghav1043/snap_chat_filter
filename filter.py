#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 02:44:12 2019

@author: raghav
"""

from model import load_cnn_model
import cv2
import numpy as np

model=load_cnn_model('model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera=cv2.VideoCapture(0)

if (camera.isOpened()== False): 
  print("Error opening video stream or file")
  
while(camera.isOpened()):
  # Capture frame-by-frame
  ret, frame = camera.read()
  frame2 = np.copy(frame)
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Detect faces
  faces = face_cascade.detectMultiScale(gray, 1.25, 6)
  
  for (x, y, w, h) in faces:
      # Grab the face
      gray_face = gray[y:y+h, x:x+w]
      color_face = frame[y:y+h, x:x+w]
      #Normalize to match the input format of the model - Range of pixel to [0, 1]
      gray_normalized = gray_face / 255
      
      # Resize it to 96x96 to match the input format of the model
      original_shape = gray_face.shape # A Copy for future reference
      face_resized = cv2.resize(gray_normalized, (96, 96), interpolation = cv2.INTER_AREA)
      face_resized_copy = face_resized.copy()
      face_resized = face_resized.reshape(1, 96, 96, 1)
      
      # Predicting the keypoints using the model
      keypoints = model.predict(face_resized)
      # De-Normalize the keypoints values
      keypoints = keypoints * 48 + 48
      # Map the Keypoints back to the original image
      face_resized_color = cv2.resize(color_face, (96, 96), interpolation = cv2.INTER_AREA)
      face_resized_color2 = np.copy(face_resized_color)
      
      points = []
      for i, co in enumerate(keypoints[0][0::2]):
          points.append((co, keypoints[0][1::2][i]))
      
      # Add FILTER to the frame
      sunglasses = cv2.imread('sunglasses.png', cv2.IMREAD_UNCHANGED)
      sunglass_width = int((points[7][0]-points[9][0])*1.1)
      sunglass_height = int((points[10][1]-points[8][1])/1.1)
      sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation = cv2.INTER_CUBIC)
      transparent_region = sunglass_resized[:,:,:3] != 0
      face_resized_color[int(points[9][1]):int(points[9][1])+sunglass_height, int(points[9][0]):int(points[9][0])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]
      
      # Add KEYPOINTS to the frame2
      for keypoint in points:
          cv2.circle(face_resized_color2, keypoint, 1, (0,255,0), 1)
          
      frame2[y:y+h, x:x+w] = cv2.resize(face_resized_color2, original_shape, interpolation = cv2.INTER_CUBIC)
      # Show the frame and the frame2
      cv2.imshow("Selfie Filters", frame)
      cv2.imshow("Facial Keypoints", frame2)    
          
  if cv2.waitKey(1) & 0xFF == ord("q"):
      break
          
camera.release()
cv2.destroyAllWindows()      
