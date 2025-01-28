#!/usr/bin/env python

"""
How to read & display an image
"""

import cv2

cap = cv2.VideoCapture('./data/flame.avi')
# print fps of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second: {0}".format(fps))

win_name = 'VideoWin'
cv2.namedWindow(win_name)

i = 0
while True:
    # read video frame-by-frame
    ret, frame = cap.read()
    if ret:
        i += 1
        cv2.imshow(win_name, frame)

        # uncomment if frame-by-frame play is needed
        # print("frame idx: {}".format(i))
        # if cv2.waitKey(0) == ord('q'):
        if cv2.waitKey(50) == ord('q'):
            break
    else:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

