import numpy
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from keras.utils import np_utils

#For reproducibility
seed = 42
numpy.random.seed(seed)

#Load as usual
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#But wait! CNNs need a 3d array [picture #][width][height]
#So do NOT reshape into a 2d array! Instead:
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
#The '1' is added for Theano to declare that depth=1 (grayscale), as opposed to depth=3 (RGB color)
#Note: shape was originally (60000, 28, 28) so you're not actually changing the array dimensions
#Just declaring depth and type!

#Normalize from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

#One hot encode from label to binary
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#Now to create the CNN model!
#Layer 1: Convolutional2D. 32 feature maps with size 5x5 and expects [picture #][width][height]
#This layer takes a filter/neuron/kernel and identifies a 5x5 vector in the image
#As this layer moves around (or "convolves") around the image, the each unique vector (there should be 32) is put into the CNN layer
#Ideally, each CNN layer would identify a "feature"
#Link: https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/
#
#Layer 2: MaxPooling2D. Pooling layer that shrinks image by 0.5 but keeps important features (lines, etc.)
#Layer 3: Dropout. Randomly excludes 20%, prevents overfitting.
#Layer 4: Flatten.
#The ReLu layer wants one big picture to process...not a bunch of small filters
#Flattening takes a vector of 2d matrices (all the filters) and squishes them into a 2d array
#that can be processed by a Dense layer
#
#Layer 5: Relu. Decides which pixels in the simplified picture are there or not.
#Less neurons = better generalizations, more neurons = more specific
#We have 128 neurons in our hidden layer right now. There's no "right" answer (yet!)
#
#Layer 6: Output. Uses softmax to classify picture.
#
#Link: https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2

def baseline_model():
	model=Sequential()
	model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(28, 28, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = baseline_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print("error: %.2f%%" % (100-scores[1]*100))

#error: 1.11%
