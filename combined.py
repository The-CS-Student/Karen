import random
import cv2
import numpy as np
import os
import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(17,GPIO.OUT)
GPIO.output(17,GPIO.LOW)
import subprocess

def play(audio_file_path):
    subprocess.call(["ffplay", "-nodisp", "-autoexit", audio_file_path])

index = 0
threshhold = 1
a = 0
cap = cv2.VideoCapture(0)
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
            face_list.append(["Unknown",confidence])

    return face_list
face_cascade = cv2.CascadeClassifier('haarCascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarCascades/haarcascade_eye_tree_eyeglasses.xml')
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
    if(a==0):
        if len(faces)==1:
            
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                fullpred = predict(gray,subjects,face_recognizer)[0]
                if(str(fullpred[0])!="Unknown"):
                    a = 1
                    GPIO.output(17,GPIO.HIGH)
                    cv2.putText(img,str(fullpred[0])+" "+str(fullpred[1]),(x,y-20),font, 
        fontScale,
        fontColor,
        lineType)
                                    
    else:
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
                    play("2.mp3")
                    index = 0
                    
                else:
                    index=0
            else:
                if(len(leftEye)==0 and len(rightEye)==0):
                    
                    index+=1
                else:
                    
                    index=0
            for (ex,ey,ew,eh) in rightEye:
                cv2.rectangle(roi_color,(ex+w//2,ey),(ex+ew+w//2,ey+eh),(255,255,0),2)
            for (ex,ey,ew,eh) in leftEye:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('ADAM',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        GPIO.output(17,GPIO.LOW)
        cap.release()
        cv2.destroyAllWindows()