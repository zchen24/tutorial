#!/usr/bin/env python3
"""
Example threading Lock

See Effective Python Item 38

1) without Lock, there is race condition
2) with Lock, it take ~30 times longer to finish counting

"""

import time
import threading


class Counter:
    def __init__(self):
        self.count = 0

    def increment(self, offset):
        self.count += offset


class LockingCounter(Counter):
    def __init__(self):
        super(LockingCounter, self).__init__()
        self.lock = threading.Lock()

    def increment(self, offset):
        with self.lock:
            self.count += offset


def worker(sensor_index, how_many, counter:Counter):
    """
    :param sensor_index: not used
    :param how_many: increment how many times
    :param counter: counter object
    :return:
    """
    for _ in range(how_many):
        counter.increment(1)


def run_threads(func, how_many, counter:Counter):
    all_threads = []
    for i in range(5):
        args = (i, how_many, counter)
        thread = threading.Thread(target=func, args=args)
        all_threads.append(thread)
        thread.start()
    for thread in all_threads:
        thread.join()



how_many = 10**5
counter = Counter()
locking_counter = LockingCounter()
t0 = time.time()
run_threads(worker, how_many, counter)
print('Counter should be {}, found {}, time {:.3f}s'.format(5*how_many,
                                                            counter.count,
                                                            time.time()-t0))

t0 = time.time()
run_threads(worker, how_many, locking_counter)
print('Counter should be {}, found {}, time {:.3f}s'.format(5*how_many,
                                                            locking_counter.count,
                                                            time.time()-t0))
