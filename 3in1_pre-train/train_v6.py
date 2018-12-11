# USAGE
# python train.py --dataset dataset --model output/fashion.model \
#	--categorybin output/category_lb.pickle --colorbin output/color_lb.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from mymodel.noelnet import NoelNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import pandas
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-a", "--agebin", required=True,
	help="path to output age label binarizer")
ap.add_argument("-g", "--genderbin", required=True,
	help="path to output gender label binarizer")
ap.add_argument("-r", "--racebin", required=True,
	help="path to output race label binarizer")
ap.add_argument("-p", "--plot", type=str, default="output",
	help="base filename for generated plots")
args = vars(ap.parse_args())

TRAINING_DIR = "D:/xrvision/XRV_projects/age_gender_ethicity_dataset/UTKface_dataset/noelnet_v5/dataset"
#debugfile("D:/xrvision/XRV_projects/age_gender_ethicity_dataset/UTKface_dataset/noelnet_v5/train_v6.py",
#          args='--dataset dataset --model output/noelnet.h5 \
#          --agebin output/age_lb.pickle --genderbin output/gender_lb.pickle \
#          --racebin output/race_lb.pickle')

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 50
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (71, 71, 3)
IMAGE_SIZE = 71

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(TRAINING_DIR)))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data, clothing age labels (i.e., shirts, jeans,
# dresses, etc.) along with the gender labels (i.e., red, blue, etc.)
data = []
ageLabels = []
genderLabels = []
raceLabels = []

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = img_to_array(image)
	data.append(image)

	# extract the clothing gender and age from the path and
	# update the respective lists
	(ageyr, gender, race) = imagePath.split(os.path.sep)[-2].split("_")
	ageLabels.append(ageyr)
	genderLabels.append(gender)
	raceLabels.append(race)

# scale the raw pixel intensities to the range [0, 1] and convert to
# a NumPy array
data = np.array(data, dtype="float") / 255.0
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

# convert the label lists to NumPy arrays prior to binarization
ageLabels = np.array(ageLabels)
genderLabels = np.array(genderLabels)
raceLabels = np.array(raceLabels)

# binarize both sets of labels
print("[INFO] binarizing labels...")
ageLB = LabelBinarizer()
#noel genderLB = LabelBinarizer()
raceLB = LabelBinarizer()

ageLabels = ageLB.fit_transform(ageLabels)
# Noel genderLabels = genderLB.fit_transform(genderLabels)
raceLabels = raceLB.fit_transform(raceLabels)


genderLB = LabelBinarizer()
genderLB.fit(np.array(genderLabels))
genderLB.classes_

genderLabels = pandas.get_dummies(genderLabels)  #noel error here.
genderLabels = genderLabels.astype('int32')
genderLabels = genderLabels.values



# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
#split = train_test_split(data, ageLabels, genderLabels, raceLabels,
#	test_size=0.2, random_state=42)
#(trainX, testX, trainAgeY, testAgeY,
#	trainGenderY, testGenderY,
#    trainRaceY, testRaceY) = split

from keras.preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.20)

train_generator = data_generator.flow_from_directory(TRAINING_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True, seed=13,
                                                     class_mode='categorical', batch_size=BS, subset="training")

validation_generator = data_generator.flow_from_directory(TRAINING_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True, seed=13,
                                                     class_mode='categorical', batch_size=BS, subset="validation")
#Noel version
H = model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=EPOCHS,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,
        callbacks= [checkpointer])

H = model.fit(train_generator,	{"age_output": trainAgeY, "gender_output": trainGenderY , "race_output": trainRaceY},
	validation_data=(testX, {"age_output": testAgeY, "gender_output": testGenderY, "race_output": testRaceY}),
	epochs=EPOCHS,
	verbose=1,
   callbacks= [checkpointer])

##########################################
checkpointer = ModelCheckpoint(filepath='./result_dir/noelnet--{epoch:03d}.h5',
                                verbose=1,
                                save_best_only=True)

#early_stopper = EarlyStopping(patience=25)

#tensorboard = TensorBoard(log_dir=os.path.join(result_dir, 'logs'),
#                          histogram_freq=0, write_graph=False,
#                          write_images=True)
 ########################################
# initialize our NoelNet multi-output network
model = NoelNet.build(71, 71,
	numAges=len(ageLB.classes_),
	numGenders=2,
	numRaces=len(raceLB.classes_),
	finalAct="softmax")

# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
	"age_output": "categorical_crossentropy",
	"gender_output": "categorical_crossentropy",
    	"race_output": "categorical_crossentropy",
}

lossWeights = {"age_output": 1.0, "gender_output": 1.0, "race_output": 1.0}

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
	metrics=["accuracy"])

# train the network to perform multi-output classification
H = model.fit(trainX,	{"age_output": trainAgeY, "gender_output": trainGenderY , "race_output": trainRaceY},
	validation_data=(testX, {"age_output": testAgeY, "gender_output": testGenderY, "race_output": testRaceY}),
	epochs=EPOCHS,
	verbose=1,
   callbacks= [checkpointer])

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the age binarizer to disk
print("[INFO] serializing age label binarizer...")
f = open(args["agebin"], "wb")
f.write(pickle.dumps(ageLB))
f.close()

# save the gender binarizer to disk
print("[INFO] serializing gender label binarizer...")
f = open(args["genderbin"], "wb")
f.write(pickle.dumps(genderLB))
f.close()

# save the race binarizer to disk
print("[INFO] serializing race label binarizer...")
f = open(args["racebin"], "wb")
f.write(pickle.dumps(raceLB))
f.close()

# plot the total loss, age loss, and gender race loss
lossNames = ["loss", "age_output_loss", "gender_output_loss", "race_output_loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(4, 1, figsize=(13, 13))

# loop over the loss names
for (i, l) in enumerate(lossNames):
	# plot the loss for both the training and validation data
	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Loss")
	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
		label="val_" + l)
	ax[i].legend()

# save the losses figure
plt.tight_layout()
plt.savefig("{}_losses.jpg".format(args["plot"]))
plt.close()

# create a new figure for the accuracies
accuracyNames = ["age_output_acc", "gender_output_acc","race_output_acc"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(8, 8))

# loop over the accuracy names
for (i, l) in enumerate(accuracyNames):
	# plot the loss for both the training and validation data
	ax[i].set_title("Accuracy for {}".format(l))
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Accuracy")
	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
		label="val_" + l)
	ax[i].legend()

# save the accuracies figure
plt.tight_layout()
plt.savefig("{}_accs.jpg".format(args["plot"]))
plt.close()