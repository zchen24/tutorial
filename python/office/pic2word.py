#!/usr/bin/env python3

"""
How to convert a series of pictures into a word document
"""

import docx
from docx.shared import Inches
from tkinter import filedialog
from tkinter import Tk
import glob
import os


root = Tk()
root.withdraw()

my_folder = filedialog.askdirectory()
print('Folder: {}'.format(my_folder))

imgs = sorted(glob.glob(os.path.join(my_folder, '*.jpg')))
print(imgs)

doc = docx.Document()

ratio = 0.68
for i in imgs:
    f_name = os.path.join(my_folder, i)
    print('f_name = {}'.format(f_name))
    doc.add_picture(os.path.join(my_folder, i),
                    width=Inches(ratio * 8.5))
    doc.add_page_break()

doc.save(os.path.join(my_folder, 'tmp.docx'))
