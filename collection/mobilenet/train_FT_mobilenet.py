from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers
from keras.preprocessing import image
from keras.applications import MobileNet
from keras.models import Model
# Importing other necessary libraries
from sklearn.metrics import classification_report,confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import h5py, os, itertools, heapq
from keras.layers import Input
import gc
import keras

# Declaring shape of input images and number of categories to classify
#input_shape = (224, 224, 3)
num_classes = 4
set_batch_size = 32
image_size = 64  # 2was 224

############################################
# Free GPU memory
model = None
del model
gc.collect()

# race model
race_path = "D:/xrvision/XRV_projects/age_gender_ethicity_dataset/UTKface_dataset/racenet_v13_abandon_aligned/\
mobilenet_race_newTL.48-0.65.h5"

from keras.utils.generic_utils import CustomObjectScope
with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    race_model = load_model(race_path)

for layer in race_model.layers:
	layer.trainable = True
	
race_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
race_model.summary()


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 40,
                                   zoom_range = 0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip = True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./dataset/train',
                                                 target_size = (image_size, image_size),
                                                 batch_size = set_batch_size,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('./dataset/test',
                                            target_size = (image_size, image_size),
                                            batch_size = set_batch_size,
                                            class_mode = 'categorical')

print(training_set.class_indices)


# Setting callbacks parameters
checkpointer = ModelCheckpoint(filepath='mobilenet_race_newFT_.{epoch:02d}-{val_loss:.2f}.h5', verbose=1, save_best_only=True)
filename='mobilenet_race.csv'
csv_log = CSVLogger(filename, separator=',', append=False)

hist = race_model.fit_generator(training_set,
                           steps_per_epoch = (17055//set_batch_size),
                           epochs = 30,
                           validation_data = test_set,
                           validation_steps = (6414//set_batch_size), 
                           workers = 4, 
                           callbacks = [csv_log, checkpointer])

model.save('mobilenet_race_FT.h5')