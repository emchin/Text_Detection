# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import pyrebase
from firebase.firebase import FirebaseApplication
import datetime
import time
import math
import numpy as np
import os
import base64
from PIL import Image
#Note: PIL is added because just downloading the library
#process the image a little. It might help. It definitely
#doesn't hurt.

#First, configure Firebase
#with the created Firebase account details
config = {
  "apiKey": "AIzaSyANGZEkND-lPaN2JXuMhyUKDmXM7KCN0n8",
  "authDomain": "test-7b06d.firebaseapp.com",
  "databaseURL": "https://test-7b06d.firebaseio.com",
  "storageBucket": "test-7b06d.appspot.com"
}

#Initialize...
firebase = pyrebase.initialize_app(config)
db = firebase.database()

#Load img from firebase
storage = firebase.storage()
imgFile = '/Users/emily/Desktop/downloaded.png'
storage.child("images/hello.jpg").download(imgFile)

#Read the image. Adding "0" makes this image grayscale
img = cv2.imread(imgFile,0)

#If you haven't given the program an image, 
#you're going to get this error:
if img is None:
	print("Could not read:", imgFile)

#Now isolate the dark text from the pale background.
#Text is now black, background is now white.
#This way, it's easy to detect the text from the picture
thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

#Let's make the grayscale image bigger!!
#Note: this makes the text detection MUCH better.
#Please do NOT delete this line!!
gray = cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

#Add a little blur to the picture
img = cv2.bilateralFilter(img,3,75,75)

#Aaaand that's all, folks!
#The image is done being processsed.
#Save final grayscale image to a new image file
filename="/Users/emily/Desktop/gray_image.png"
cv2.imwrite(filename, gray)

#Save the text from the image as a variable "text"
#Do we need this? I seriously hope we do...
text = pytesseract.image_to_string(Image.open(filename), lang = 'eng')

print(text)

#Show the image; it will go away if you hit the "0" key
#cv2.imshow("Output", gray)
#cv2.waitKey(0)

#Convert image "gray" to string in base64
with open(filename, "rb") as imageFile:
	gray_string = base64.b64encode(imageFile.read())
	imageFile.close()

#There are two children in the database
#"data" and "photos"
posts_ref = db.child("data")

time_now = time.time()

#db.push({"time": time_now, "text": text})

#Push to firebase
url = "https://test-7b06d.firebaseio.com"
fb = FirebaseApplication(url, None)
result = fb.patch(url+"/data/" + "derived_from_hello1", {"text":text, "time":time.time()})

#Yayy! We uploaded the time, text and image to the
#firebase server!
#Delete the file so that when the next person uploads a pic
#we don't have a "same file name in same place" error
os.remove(filename)
os.remove("/Users/emily/Desktop/downloaded.png")
