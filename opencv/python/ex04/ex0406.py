#!/usr/bin/env python

import cv2
import numpy as np


img = np.zeros((400, 600, 3), dtype=np.uint8)
is_abort = False


def callback_mouse_edit(event, x, y, flags, param):
    global img
    global is_abort

    img_show = np.copy(img)
    if event == cv2.EVENT_LBUTTONDOWN:
        text = ""
        while True:
            key = cv2.waitKey(10)
            if key == 27:
                print("Esc key pressed")
                cv2.imshow("window", img)
                is_abort = True
                break
            elif key == 10:
                print("Enter key pressed")
                img = np.copy(img_show)
                break
            else:
                # key text label
                if key == 255:
                    continue
                # backspace
                if key == 8:
                    if text:
                        text = text[:-1]
                else:
                    text = text + chr(key)                    
                    print("key = {}".format(key))
                img_show = np.copy(img)
                cv2.putText(img_show, text, (x, y),
                            cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255))
                cv2.imshow("window", img_show)
                

win_name = "window"
cv2.namedWindow(win_name)
cv2.setMouseCallback(win_name, callback_mouse_edit, param=None)
cv2.imshow("window", img)
cv2.startWindowThread()
while not is_abort:
    pass
cv2.destroyAllWindows()


