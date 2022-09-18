#!/usr/bin/env python3

"""

Date: 2022-09-18
"""


from openpyxl import Workbook
import glob, os


fs = glob.glob("*.py")
wb = Workbook()
ws = wb.active
ws.append(['File', 'Type', 'Size(bytes)'])
for f in fs:
    ext = os.path.splitext(f)[-1][1:]
    f_size_KB = os.path.getsize(f)
    ws.append([f, ext, f_size_KB])

wb.save('tmp.xlsx')