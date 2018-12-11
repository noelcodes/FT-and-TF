from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, ZeroPadding2D , Dense, Dropout, Flatten, Activation
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ModelCheckpoint, CSVLogger

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.summary()

model.load_weights('vgg_face_weights.h5') 

model.add(Dense(units = 4, activation = 'softmax'))  # 13 classes
for layer in model.layers[:-1]:
	layer.trainable = False
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()


set_image_size = 224
set_batch_size = 64
no_of_epochs = 50

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 40,
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
checkpointer = ModelCheckpoint(filepath='race_vggface_224.{epoch:02d}-{val_loss:.2f}.h5', verbose=1, save_best_only=True)
filename='model_race.csv'
csv_log = CSVLogger(filename, separator=',', append=False)

hist = model.fit_generator(training_set,
                           steps_per_epoch = (17056//set_batch_size),
                           epochs = no_of_epochs,
                           validation_data = test_set,
                           validation_steps = (6414//set_batch_size), 
                           workers = 4, 
                           callbacks = [csv_log, checkpointer])

model.save('model_race.h5')