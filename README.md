# Text_Detection

ocr.py uses pytesseract's method "image_to_string" to parse a picture. We used this in the 2018-2019 Tinovation project to identify text from a book's page.  This program turns the image grayscale and identifies the blur to increase accuracy of the OCR. Once the text is identified, it is converted into binary and sent (with a timestamp) to a Firebase database.
