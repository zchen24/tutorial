#!/usr/bin/env python

"""
Read & display a video controlled by sliders
1) control the position within the video
2) binary slider play/pause (label appropriately)
"""

import cv2


# video capture
cap = cv2.VideoCapture('../data/flame.avi')
g_play = True


def callback_pos_trackbar(pos):
    global cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)


def callback_play_trackbar(pos):
    global g_play
    if pos == 1:
        g_play = True
    elif pos == 0:
        g_play = False
    else:
        print('Unsupported button position')



total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

win_name = 'VideoWin'
cv2.namedWindow(win_name)
cv2.createTrackbar('Pos', win_name, 0, total_frames, callback_pos_trackbar)
cv2.createTrackbar('Play', win_name, 0, 1, callback_play_trackbar)
cv2.setTrackbarPos('Play', win_name, 1)


while True:
    if g_play:
        ret, img = cap.read()
        frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.setTrackbarPos('Pos', win_name, frame_pos)

        if not ret:
            cv2.waitKey(30)
        else:
            cv2.imshow(win_name, img)
            k = cv2.waitKey(30)
            if (k & 0xFF) == ord('q'):
                break
    else:
        cv2.waitKey(30)

cv2.destroyAllWindows()