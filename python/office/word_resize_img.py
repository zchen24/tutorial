#!/usr/bin/env python3

"""
How to resize an image in a Word document

Author: Zihan Chen
Date: 2024-03-20
"""

from docx import Document
from docx.shared import Inches

# Load the .docx file
doc = Document('tmp.docx')

for p in doc.paragraphs:
    print(p.text, "len run = ", len(p.runs))
    r = p.runs[0]

    # 判断是图片
    if r._element.xml.find('w:drawing') != -1:
        print("found a drawing")
        ele = r._element
        tmp = ele.xpath('./w:drawing/wp:inline/wp:extent')
        cx_emus = int(tmp[0].get('cx'))
        cy_emus = int(tmp[0].get('cy'))

        cx_set_in = 2
        cx_set_emus = Inches(cx_set_in)
        cy_set_emus = int(cx_set_emus * cy_emus / cx_emus)
        tmp[0].set('cx', str(cx_set_emus))
        tmp[0].set('cy', str(cy_set_emus))

        tmp = ele.xpath('*//pic:spPr/a:xfrm/a:ext')
        tmp[0].set('cx', str(cx_set_emus))
        tmp[0].set('cy', str(cy_set_emus))

        print("cx = ", tmp[0].get('cx'), "cy = ", tmp[0].get('cy'))

# Save the modified document
doc.save('tmp_small.docx')
