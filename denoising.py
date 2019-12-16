import numpy as np
import cv2
from matplotlib import pyplot as plt

#image
img = cv2.imread('/Users/emily/Desktop/website.png') 
#convert to rgb
b,g,r = cv2.split(img)
rgb_img = cv2.merge([r,g,b])

# Denoising
# Parameters: Source image, idk, Denoising strength, Color denoising strength, Template size(odd, 7 recommended), Search size(odd, 21 recommended)
dst = cv2.fastNlMeansDenoisingColored(img,None,12,10,7,21)
#convert to rgb
b,g,r = cv2.split(dst)
img = cv2.merge([r,g,b])

#plt.subplot(211),plt.imshow(rgb_img)
plt.imshow(rgb_dst)
cv2.imwrite("/Users/emily/Desktop/denoised.png", rgb_dst)
plt.show()
