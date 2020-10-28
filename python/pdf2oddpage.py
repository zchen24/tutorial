#!/usr/bin/env python3

"""
Export odd pages of a PDF file
"""

from PyPDF2 import PdfFileWriter, PdfFileReader
from tkinter import filedialog
from tkinter import Tk

root = Tk()
root.withdraw()

my_file = filedialog.askopenfilename()
print('File: {}'.format(my_file))

reader = PdfFileReader(my_file)
num_pages = reader.getNumPages()
writer = PdfFileWriter()

for i in range(0, num_pages, 2):
    page = reader.getPage(i)
    writer.addPage(page)

with open('odd.pdf', 'wb') as output:
    writer.write(output)
print('Odd pages have been written to odd.pdf')
