from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers
from keras.preprocessing import image
from keras.applications import ResNet50
from keras.models import Model
# Importing other necessary libraries
from sklearn.metrics import classification_report,confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import h5py, os, itertools, heapq
from keras.layers import Input
from keras.optimizers import SGD
import gc

# Declaring shape of input images and number of categories to classify
#input_shape = (224, 224, 3)
num_classes = 4
set_batch_size = 32
set_image_size = 200

#image_input = Input(shape=(set_image_size, set_image_size, 3))
#model = ResNet50(input_tensor=image_input, include_top=True,weights="imagenet")
#last_layer = model.get_layer('avg_pool').output
#x= Flatten(name='flatten')(last_layer)
#out = Dense(num_classes, activation='softmax', name='output_layer')(x)
#custom_resnet_model = Model(inputs=image_input,outputs= out)

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

# Free GPU memory
model = None
del model
gc.collect()

model_path = "D:/xrvision/XRV_projects/age_gender_ethicity_dataset/\
UTKface_dataset/racenet_v12_resnet_mcci_less_pretrain/\
model_race.29-0.24.h5"

print("[INFO] loading network...")
model = load_model(model_path)

for layer in model.layers:
	layer.trainable = True

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

#custom_resnet_model.layers[-6].trainable
#custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#custom_resnet_model.summary()

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 20,
                                   zoom_range = 0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip = True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./dataset/train',
                                                 target_size = (set_image_size, set_image_size),
                                                 batch_size = set_batch_size,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('./dataset/test',
                                            target_size = (set_image_size, set_image_size),
                                            batch_size = set_batch_size,
                                            class_mode = 'categorical')

print(training_set.class_indices)

# Setting callbacks parameters
checkpointer = ModelCheckpoint(filepath='model_race_pretrain_ADAM.{epoch:02d}-{val_loss:.2f}.h5', verbose=1, save_best_only=True)
filename='model_race.csv'
csv_log = CSVLogger(filename, separator=',', append=False)

hist = model.fit_generator(training_set,
                           steps_per_epoch = (14970//set_batch_size),
                           epochs = 5000,
                           validation_data = test_set,
                           validation_steps = (6414//set_batch_size), 
                           workers = 4, 
                           callbacks = [csv_log, checkpointer])

model.save('model_race.h5')





















