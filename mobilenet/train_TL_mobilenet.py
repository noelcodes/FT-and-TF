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
from keras.optimizers import SGD

# Declaring shape of input images and number of categories to classify
#input_shape = (224, 224, 3)
num_classes = 4
set_batch_size = 128
image_size = 64  # 2was 224

input_tensor = Input(shape=(image_size, image_size, 3))
base_model = MobileNet(input_tensor=input_tensor, include_top=False, weights='imagenet')

for layer in base_model.layers[:]:
	layer.trainable = False
    
x = base_model.output
x = Flatten()(x)
x = Dense(20, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(10, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(num_classes, activation='softmax', name='predictions')(x)
mobilenet = Model(inputs=base_model.input, outputs=x)
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
mobilenet.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

mobilenet.summary()


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
checkpointer = ModelCheckpoint(filepath='mobilenet_race_newTL.{epoch:02d}-{val_loss:.2f}.h5', verbose=1, save_best_only=True)
filename='mobilenet_race.csv'
csv_log = CSVLogger(filename, separator=',', append=False)

hist = mobilenet.fit_generator(training_set,
                           steps_per_epoch = (17055//set_batch_size),
                           epochs = 50,
                           validation_data = test_set,
                           validation_steps = (6414//set_batch_size), 
                           workers = 4, 
                           callbacks = [csv_log, checkpointer])

model.save('mobilenet_race.h5')