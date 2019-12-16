import os
from PIL import Image
import pytesseract
import cv2 as dankmemes
import cv2
import numpy as np
from PIL import Image

from symspellpy.symspellpy import SymSpell, Verbosity  # import the module

def main():

        #Load the image from the desktop
    imgFile = '/Users/emily/Desktop/basic_word2.png'

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
    gray = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

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

    # maximum edit distance per dictionary precalculation
    max_edit_distance_dictionary = 5
    prefix_length = 7
    # create object
    sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
    # load dictionary
    dictionary_path = os.path.join(os.path.dirname(r"/Users/emily/Documents/Tinovation/spellcheck2.py"),
                                   "/Users/emily/Desktop/frequency_dictionary_en_82_765.txt")
    term_index = 0  # column of the term in the dictionary text file
    count_index = 1  # column of the term frequency in the dictionary text file
    if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
        print("Dictionary file not found")
        return

#     # lookup suggestions for single-word input strings
#     input_term = "memebers"  # misspelling of "members"
#     # max edit distance per lookup
#     # (max_edit_distance_lookup <= max_edit_distance_dictionary)
#     max_edit_distance_lookup = 2
#     suggestion_verbosity = Verbosity.CLOSEST  # TOP, CLOSEST, ALL
#     suggestions = sym_spell.lookup(input_term, suggestion_verbosity,
#                                    max_edit_distance_lookup)
#     # display suggestion term, term frequency, and edit distance
#     for suggestion in suggestions:
#         print("{}, {}, {}".format(suggestion.term, suggestion.distance,
#                                   suggestion.count))

#     # lookup suggestions for multi-word input strings (supports compound
#     # splitting & merging)
    input_term = ("whereis th elove hehad dated forImuch of thepast who "
                  "couqdn'tread in sixtgrade and ins pired him")
    input_term = ("ront tshi liptop si ocol")
    input_term = text
    # max edit distance per lookup (per single word, not per whole input string)
    max_edit_distance_lookup = 2
    suggestions = sym_spell.lookup_compound(input_term,
                                            max_edit_distance_lookup)
    # display suggestion term, edit distance, and term frequency
    for suggestion in suggestions:
        print("{}, {}, {}".format(suggestion.term, suggestion.distance,
                                  suggestion.count))

if __name__ == "__main__":
    main()