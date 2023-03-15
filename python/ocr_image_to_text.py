#!/usr/bin/env python3

"""
A script that extracts text from an image

Author: ChatGPT
Date: 2023-03-16
"""

import pytesseract
from PIL import Image
import argparse

def ocr_image(image_path):
    # Open the image
    image = Image.open(image_path)

    # Perform OCR on the image
    text = pytesseract.image_to_string(image, lang='chi_sim')

    return text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Image path')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-i', '--image', type=str,
                       help='image path')
    args = parser.parse_args()

    # "input_image.jpg"  # Replace with your image file path
    image_path = args.image

    # Perform OCR on the image and print the extracted text
    extracted_text = ocr_image(image_path)
    print("Extracted text from the image:")
    print(extracted_text)
