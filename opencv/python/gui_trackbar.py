#!/usr/bin/env python

"""
Shows how to use HIGHGUI trackbar
- slider to follow video frame
- change the frame pos when slider is moved
- get/set capture properties
"""

import cv2

# global cap
cap = cv2.VideoCapture('./data/flame.avi')


def on_position_trackbar(pos):
    global cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)


total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

win_name = 'VideoWin'
cv2.namedWindow(win_name)
cv2.createTrackbar('P', win_name, 0, total_frames, on_position_trackbar)

while True:
    # read video frame-by-frame
    ret, frame = cap.read()
    frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos('P', win_name, frame_pos)

    if ret:
        cv2.imshow(win_name, frame)
        if cv2.waitKey(50) == ord('q'):
            break
    else:
        # end of file, press any key
        cv2.waitKey(0)
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

