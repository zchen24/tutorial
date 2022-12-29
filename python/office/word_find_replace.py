#!/usr/bin/env python3

"""
How to find and replace text in a Word document

Author: Zihan Chen
Date: 2022-12-29
"""

import docx

def find_replace(paragraph_keyword, draft_keyword, paragraph):
    if paragraph_keyword in paragraph.text:
        # print("found")
        paragraph.text = paragraph.text.replace(paragraph_keyword, draft_keyword)

doc = docx.Document('test.docx')

for pa in doc.paragraphs:
    find_replace('KEYWORD', "1122334455-UPDATED", pa)

doc.save('test-updated.docx')
