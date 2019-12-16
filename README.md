# Text_Detection
### This was the 2018-2019 project for the Python team in Tinovation.

The 2017-2018 "Restore" project hoped to restore an image of a blurry, torn or otherwise compromised text. The photo taken would be uploaded via app to a Firebase database. As the Python team, our objective was to (a) retrieve this image from the Firebase database; (b) parse the picture, applying filters to make the image more decipherable; (c) use machine learning to identify the characters and text in the image; and (d) send the text and other relevant information back to Firebase for the other teams to retreive.

We successfully completed all of our goals.

Our first attempt, "ocr.py", uses pytesseract's given method "image_to_string" to perform OCR a picture. We used this to identify text from a book's page.  This program turns the image grayscale and identifies the blur to increase accuracy of the OCR. Once the text is identified, it is converted into binary and sent (with a timestamp) to a Firebase database.  This program does **not** use Machine Learning.

The first attempt did successfully read and write to the Firebase database; however, it wasn't perfect. While the "image_to_string" via pytesseract was simple, it was also not very accurate and it could not differentiate between different columns, etc. Furthermore, when we used "image_to_string", the program only identified characters, and thus necessitated a spellcheck program ("spellcheck.py", "spellcheck2.py") to correct the typos and misspellings of the program.

Keeping this in mind, we decided to continue working on our project to improve the accuracy of our program's OCR capabilities.

MNIST_basic.py uses a small NN model with two Dense layers to identify characters. This model **does** use ML and has an error rate of 1.82%.

MNIST_CNN.py uses a larger CNN model that includes a Conv2D layer (convolutional layer), MaxPooling2D, Dropout, Flatten, and one final Dense layer. This model **does** use ML and has an error rate of 1.11%.

MNIST_DNN.py uses the largest model that uses a Conv2D layer, MaxPooling2D, *another* Conv2D layer, *another* MaxPooling2D, Dropout, Flatten, and two Dense layers. This model **does** use ML and has an error rate of 0.89%.

This project tackles Firebase data storage and retrieval, image processing, pytesseract, and conversion between image and binary data storage. In the realm of Machine Learning, the project uses deep/convolutional neural networks, keras and TensorFlow to address concepts such as convolutional layers, activation functions, loss functions, training versus testing data.

### Tinovation 2018-2019 Resources:
Throughout the year, we also learned valuable skills such as the basics of numpy, basic ML concepts and applications via iris and boston housing databases. Links are below:

Iris Analysis: https://tinyurl.com/tinovationiris

Boston Housing Analysis: https://tinyurl.com/bostonhousing
