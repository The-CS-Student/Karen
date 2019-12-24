import numpy as np
import cv2
import time
import random
import serial
ser1 = serial.Serial('/dev/ttyACM0', 9600)
face_cascade = cv2.CascadeClassifier('haarCascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarCascades/haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)


index = 0
threshhold = 2

    
while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	if len(faces)>0:
		
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_grayLeft = gray[y:y+h, x:x+w//2]
			roi_grayRight = gray[y:y+h, x+w//2:x+w]
			roi_color = img[y:y+h, x:x+w]
			leftEye = eye_cascade.detectMultiScale(roi_grayLeft)
			rightEye = eye_cascade.detectMultiScale(roi_grayRight) 
			if(index>threshhold):
				if(len(leftEye)==0 and len(rightEye)==0):
					ser1.write('s'.encode())
					index = 0
					
				else:
					ser1.write('n'.encode())
					index=0
			else:
				if(len(leftEye)==0 and len(rightEye)==0):
					ser1.write('n'.encode())
					index+=1
				else:
					ser1.write('n'.encode())
					index=0
			for (ex,ey,ew,eh) in rightEye:
				cv2.rectangle(roi_color,(ex+w//2,ey),(ex+ew+w//2,ey+eh),(255,255,0),2)
			for (ex,ey,ew,eh) in leftEye:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	cv2.imshow('Drowsiness Detecter',img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		cap.release()
		cv2.destroyAllWindows()

