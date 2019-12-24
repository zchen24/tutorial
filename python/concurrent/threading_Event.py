#!/usr/bin/env python3
"""
Example threading Event
"""

import threading
import time


def worker(stop_event:threading.Event):
    t_start = time.time()
    while not stop_event.is_set():
        print('{:.03f}'.format(time.time()-t_start))
        time.sleep(0.1)


if __name__ == '__main__':
    stop_event = threading.Event()
    t = threading.Thread(target=worker, args=[stop_event])
    t.start()
    time.sleep(3.0)
    stop_event.set()
    t.join()
