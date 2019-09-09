# Text_Detection

ocr.py uses pytesseract's method "image_to_string" to parse a picture. We used this in the 2018-2019 Tinovation project to identify text from a book's page.  This program turns the image grayscale and identifies the blur to increase accuracy of the OCR. Once the text is identified, it is converted into binary and sent (with a timestamp) to a Firebase database.  This program does **not** use Machine Learning.


MNIST_basic.py uses a small NN model with two Dense layers to identify characters. This model **does** use ML and has an error rate of 1.82%.

MNIST_CNN.py uses a larger CNN model that includes a Conv2D layer (convolutional layer), MaxPooling2D, Dropout, Flatten, and one final Dense layer. This model **does** use ML and has an error rate of 1.11%.

MNIST_DNN.py uses the largest model that uses a Conv2D layer, MaxPooling2D, *another* Conv2D layer, *another* MaxPooling2D, Dropout, Flatten, and two Dense layers. This model **does** use ML and has an error rate of 0.89%

# Tinovation 2018-2019 Resources

Iris Analysis: https://tinyurl.com/tinovationiris

Boston Housing Analysis: https://tinyurl.com/bostonhousing
