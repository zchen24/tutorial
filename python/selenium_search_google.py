#!/usr/bin/env python

"""
Setup:
1) pip install selenium
2) Download chromedriver online
3) Add chromedriver to PATH

Demo:
1) open chrome
2) search local weather
3) close chrome after 3 seconds

How to find element:
Use browser's "inspect element" feature, copy selector
"""

from __future__ import print_function
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


if __name__ == '__main__':
    print('auto search google')

    driver = webdriver.Chrome()
    driver.get("https://www.google.com/")
    assert "Google" in driver.title

    ele = driver.find_element_by_css_selector('#lst-ib')
    ele.send_keys('local weather')
    ele.send_keys(Keys.RETURN)
    time.sleep(3)
    driver.close()
