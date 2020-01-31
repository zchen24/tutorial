#!/usr/bin/env python

"""
How to write to a video file
"""

import cv2


cap = cv2.VideoCapture('/dev/video0')
_, frame = cap.read()
img_size = frame.shape[:2][::-1]
fps = 30

writer = cv2.VideoWriter('video.avi',
                         cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'),
                         fps,
                         img_size)

while True:
    # read video frame-by-frame
    ret, frame = cap.read()
    if ret:
        writer.write(frame)
        cv2.imshow('Preview', frame)
        if cv2.waitKey(10) == ord('q'):
            break
    else:
        break


cv2.destroyAllWindows()
cap.release()
writer.release()

