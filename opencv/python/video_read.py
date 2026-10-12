#!/usr/bin/env python

"""
How to read & display a video frame by frame

Usage: python video_read.py [video_file]

Keys:  right arrow / any other key = next frame
       left arrow                  = previous frame
       q / Esc                     = quit

Recently shown frames are kept in a small JPEG cache so stepping back is
instant; stepping back beyond the cache falls back to a (slow) decoder seek.
"""

import sys
from collections import OrderedDict
import cv2

LEFT_KEYS = {63234, 2, 65361, 81, 2424832}     # macOS Cocoa, GTK/Qt, Windows variants
RIGHT_KEYS = {63235, 3, 65363, 83, 2555904}
QUIT_KEYS = {ord('q'), 27}

CACHE_MAX = 400          # recently decoded frames kept as JPEG

path = sys.argv[1] if len(sys.argv) > 1 else './data/flame.avi'
cap = cv2.VideoCapture(path)
# print fps of the video
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Frames per second: {0}, frames: {1}, size: {2}x{3}".format(fps, n_frames, width, height))

win_name = 'VideoWin'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
if width > 1920:         # shrink the window for 4K sources, image is unchanged
    cv2.resizeWindow(win_name, 1920, int(height * 1920 / width))

cache = OrderedDict()    # frame index (0-based) -> jpeg bytes
stream_pos = 0           # index of the frame the next cap.read() returns


def get_frame(idx):
    """Return frame idx (0-based) or None. Uses sequential read, cache, or seek."""
    global stream_pos
    if idx in cache:
        cache.move_to_end(idx)
        return cv2.imdecode(cache[idx], cv2.IMREAD_COLOR)
    if idx != stream_pos:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)      # slow path (re-decodes from keyframe)
        stream_pos = idx
    ret, frame = cap.read()
    if not ret:
        return None
    stream_pos += 1
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        cache[idx] = buf
        if len(cache) > CACHE_MAX:
            cache.popitem(last=False)
    return frame


i = 0
while True:
    frame = get_frame(i)
    if frame is None:
        if i > 0:
            i -= 1          # ran past the end: stay on the last valid frame
            continue
        break
    label = "frame {}  t = {:.1f} ms".format(i + 1, i * 1000.0 / fps if fps else 0)
    scale = max(0.6, frame.shape[1] / 1536.0)
    cv2.putText(frame, label, (20, int(36 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), int(4 * scale) + 1)
    cv2.putText(frame, label, (20, int(36 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), max(1, int(1.5 * scale)))
    cv2.imshow(win_name, frame)
    print("frame idx: {}".format(i + 1))

    key = cv2.waitKeyEx(0)
    if key in QUIT_KEYS:
        break
    elif key in LEFT_KEYS:
        i = max(0, i - 1)
    else:                   # right arrow or any other key
        i = min(n_frames - 1, i + 1) if n_frames > 0 else i + 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
