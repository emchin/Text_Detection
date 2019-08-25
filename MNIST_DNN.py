import numpy
from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from keras.utils import np_utils

seed = 42
numpy.random.seed(seed)

#load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

#normalize
X_train = X_train / 255
X_test = X_test / 255

#one hot encode
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#As compared to the CNN, this has an additional convolutional + pooling
#This also has an additional fully-connected layer.
#
#Layer 1: Convolutional 2D with size 5x5 (30 feature maps)
#Layer 2: MaxPooling2D that takes "max" of 2x2 patches
#Layer 3: Convolutional 2D with size 3x3 (15 feature maps)
#Layer 4: Pooling layer that takes "max" of 2x2 patches (again)
#Layer 5: Dropout layer (20%)
#Layer 6: Flatten layer
#Layer 7: Dense layer (128 neurons)
#Layer 8: Dense layer (50 neurons)
#Layer 9: Output layer

def larger_model():
	model = Sequential()
	model.add(Conv2D(30, kernel_size=(5,5), input_shape=(28,28,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(15, kernel_size=(3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = larger_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

#Analysis
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

#Large CNN Error: 0.89%