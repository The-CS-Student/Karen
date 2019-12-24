import tensorflow as tf
import pickle
import random
import glob
import cv2
import numpy as np
import os
from keras.applications.mobilenet import preprocess_input
from keras.applications.mobilenet import decode_predictions
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
cap = cv2.VideoCapture(2)
def predict(test_img,subjects,face_recognizer):
	img = test_img.copy()

	face_list=[]

	for i in range(len(faces)):
		rect = faces[i]
		(x,y,w,h) = rect
		face = gray[y:y+w, x:x+h]

		label, confidence = face_recognizer.predict(face)

		if(confidence<70):

			face_list.append([subjects[label],confidence])
		else:
			face_list.append(["Unknown"+str(subjects[label]),confidence])

	return face_list
face_cascade = cv2.CascadeClassifier('haarCascades/haarcascade_frontalface_default.xml')
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
shapeImg = (100,100)
subjects = ['','Niranjan','Robert Downey Jr','Taylor Swift']
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("lbp7.yml")

while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	if len(faces)==1:
		
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			fullpred = predict(gray,subjects,face_recognizer)[0]

			cv2.putText(img,str(fullpred[0])+" "+str(fullpred[1]),(x,y-20),font, 
    fontScale,
    fontColor,
    lineType)
	
	cv2.imshow('Face Detecter',img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		cap.release()
		cv2.destroyAllWindows()