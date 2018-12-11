# USAGE
# python classify.py --model output/fashion.model \
#	--agebin output/age_lb.pickle --genderbin output/gender_lb.pickle \
#	--image examples/black_dress.jpg

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf
import numpy as np
import argparse
import imutils
import pickle
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-a", "--agebin", required=True,
	help="path to output age label binarizer")
ap.add_argument("-g", "--genderbin", required=True,
	help="path to output gender label binarizer")
ap.add_argument("-r", "--racebin", required=True,
	help="path to output race label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
output = imutils.resize(image, width=400)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# pre-process the image for classification
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network from disk, followed
# by the age and gender label binarizers, respectively
print("[INFO] loading network...")
model = load_model(args["model"], custom_objects={"tf": tf})
ageLB = pickle.loads(open(args["agebin"], "rb").read())
genderLB = pickle.loads(open(args["genderbin"], "rb").read())
raceLB = pickle.loads(open(args["racebin"], "rb").read())

# classify the input image using Keras' multi-output functionality
print("[INFO] classifying image...")
(ageProba, genderProba, raceProba) = model.predict(image)

# find indexes of both the age and gender outputs with the
# largest probabilities, then determine the corresponding class
# labels
ageIdx = ageProba[0].argmax()
genderIdx = genderProba[0].argmax()
raceIdx = raceProba[0].argmax()

ageLabel = ageLB.classes_[ageIdx]
genderLabel = genderLB.classes_[genderIdx]
raceLabel = raceLB.classes_[raceIdx]

# draw the age label and gender label on the image
ageText = "age: {} ({:.2f}%)".format(ageLabel,
	ageProba[0][ageIdx] * 100)
genderText = "gender: {} ({:.2f}%)".format(genderLabel,
	genderProba[0][genderIdx] * 100)
raceText = "race: {} ({:.2f}%)".format(raceLabel,
	raceProba[0][raceIdx] * 100)

cv2.putText(output, ageText, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)
cv2.putText(output, genderText, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)
cv2.putText(output, raceText, (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)
# display the predictions to the terminal as well
print("[INFO] {}".format(ageText))
print("[INFO] {}".format(genderText))
print("[INFO] {}".format(raceText))
# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)



import numpy as np
import cv2

face_cascade =  cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)
        
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
	
	
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()