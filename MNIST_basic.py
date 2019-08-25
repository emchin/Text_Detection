import numpy
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

#For reproducibility
seed = 42
numpy.random.seed(seed)

#Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#The data is in a 3d array [picture #][width][height]
#Multiply width * height to make a 2d array [pic #][pixel #]
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

#Normalize greyscale pixel value to a value from 0-1
X_train = X_train/255
X_test = X_test/255

#One hot encode the output variables ("this is a nine")
#to a binary matrix value that represents each variable (0110)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#Determine how many classes there are in our model
num_classes = y_test.shape[1]

#Baseline Model (#1)
def baseline_model():
	model = Sequential() 
	#can add layers to this model
	#
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	#num_pixels = output size dimensions (for next layer)
	#input_dim = receives arrays of (*, num_pixels), only needed in first layer
	#kernel_initializer = kernel weight matrix initializes as random/normal
	#activation = activation function is a relu graph; for each pixel, you get 0 or 1 (black or not black)
	#
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	#num_classes = output size dimensions (for # of classes)
	#no input_dim because we get input from previous Dense layer
	#activation = softmax takes the output (% likelihood for each class) and classifies
	#
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	#'categorical crossentropy' in Keras = logarithmic loss function
	#optimizer = ADAM, a gradient descent algorithm to determine weights quickly and ~accurately
	return model

model = baseline_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
#epochs = how many iterations/times the model will run through a batch of data
#batch_size = how many images the model processes per epoch
#verbose = 2 means that we want to see a one-line status detailing each epoch's results
#
scores=model.evaluate(X_test, y_test, verbose=0)
#scores = # correct when comparing to x_test and y_test
#verbose = 0 means that you don't have to show status on this step
#
print("Baseline error: %.2f%%" % (100-scores[1]*100))
#Baseline error: 1.82%