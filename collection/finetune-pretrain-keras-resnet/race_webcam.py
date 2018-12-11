from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

def adjust_gamma(image, gamma):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

main_path = "D:/xrvision/XRV_projects/age_gender_ethicity_dataset/UTKface_dataset/racenet_v12_resnet_mcci_less_pretrain/"
weights_path = main_path + "model_race.29-0.24.h5"

print("[INFO] loading network...")
model = load_model(weights_path)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
img_width, img_height = 200, 200      


face_cascade =  cv2.CascadeClassifier(main_path + 'data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5)
    cl = clahe.apply(gray)
#    cv2.imshow('cl',cl)
    #mtcnn
    
    faces = face_cascade.detectMultiScale(cl,scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:

        
        roi_color = frame[y:y+h, x:x+w]
        
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)

        blur = cv2.GaussianBlur(roi_color,(5,5),0)
        roi_color = cv2.addWeighted(blur,1.5,roi_color,-0.5,0)
        roi_color = adjust_gamma(roi_color, gamma=0.8)

        img = cv2.resize(roi_color, (img_width, img_height)) 
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255. 

        img = np.vstack([img])
        classes = model.predict(img, batch_size=1)
        pred = np.argmax(classes,axis=1)
        
        label = ["Caucasian", "Chinese", "Indian", "Malay"]
        show_pred = label[pred[0]]

 #       label = "{}: {:.2f}% ".format(label, classes[pred[0]] * 100)

        cv2.putText(frame, show_pred, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2)
	
	
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()