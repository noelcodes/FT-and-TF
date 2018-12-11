# import the necessary packages
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG19

class NoelNet:
	@staticmethod
	def build_age_branch(inputs, numAges,
		finalAct="softmax", chanDim=-1):
		a = VGG19(include_top=False)(inputs)
		a = Flatten()(a)
		a = Dense(256)(a)
		a = Activation("relu")(a)
		a = BatchNormalization()(a)
		a = Dropout(0.5)(a)
		a = Dense(numAges)(a)
		a = Activation(finalAct, name="age_output")(a)

		return a
    
	@staticmethod
	def build_gender_branch(inputs, numGenders, finalAct="softmax",chanDim=-1):
		# CONV => RELU => POOL
		g = VGG16(include_top=False)(inputs)

		g = Flatten()(g)
		g = Dense(128)(g)
		g = Activation("relu")(g)
		g = BatchNormalization()(g)
		g = Dropout(0.5)(g)
		g = Dense(numGenders)(g)
		g = Activation(finalAct, name="gender_output")(g)

		# return the gender prediction sub-network
		return g

	@staticmethod
	def build_race_branch(inputs, numRaces,
		finalAct="softmax", chanDim=-1):
		r = Xception(include_top=False)(inputs)
		
		r = Flatten()(r)
		r = Dense(256)(r)
		r = Activation("relu")(r)
		r = BatchNormalization()(r)
		r = Dropout(0.5)(r)
		r = Dense(numRaces)(r)
		r = Activation(finalAct, name="race_output")(r)

		# return the age prediction sub-network
		return r
	@staticmethod
	def build(width, height, numAges, numGenders, numRaces,
		finalAct="softmax"):
		# initialize the input shape and channel dimension (this code
		# assumes you are using TensorFlow which utilizes channels
		# last ordering)
		inputShape = (height, width, 3)
		chanDim = -1

		# construct both the "age" and "gender" sub-networks
		inputs = Input(shape=inputShape)
		ageBranch = NoelNet.build_age_branch(inputs,
			numAges, finalAct=finalAct, chanDim=chanDim)
		genderBranch = NoelNet.build_gender_branch(inputs,
			numGenders, finalAct=finalAct, chanDim=chanDim)
		raceBranch = NoelNet.build_race_branch(inputs,
			numRaces, finalAct=finalAct, chanDim=chanDim)
		# create the model using our input (the batch of images) and
		# two separate outputs -- one for the clothing age
		# branch and another for the gender branch, respectively
		model = Model(
			inputs=inputs,
			outputs=[ageBranch, genderBranch, raceBranch],
			name="noelnet")

		# return the constructed network architecture
		return model