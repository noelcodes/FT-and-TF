# gender_final\dataset\train\male

import glob
import cv2
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib  
import os
import tqdm
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=200)

os.chdir("C:/xrvision/racenet_v13_abandon_aligned/dataset/train/Malay")
new_path = "C:/xrvision/racenet_v13_abandon_aligned/dataset/train/Malay_c/"


for old_file in glob.glob("*.jpg"): 
    frame = cv2.imread(old_file)
    floatframe = frame.astype(float)
    mean_frame = np.mean(floatframe)
    div_frame = floatframe / mean_frame
    final1 = div_frame * 128
    np.clip(final1, 0, 255, out=final1)
    img = final1.astype(np.uint8)
    
    img1 = imutils.resize(img, width=600)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        	(x, y, w, h) = rect_to_bb(rect)
        	faceAligned = fa.align(img1, gray, rect)
    print(old_file)
    cv2.imwrite(new_path + old_file, faceAligned)
 
            