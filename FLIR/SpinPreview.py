#!/usr/bin/env python3

"""
Preview Video Stream using OpenCV
"""

import PySpin
import cv2
import queue
import threading
import time


def worker_record(img_queue: queue.Queue, stop_event: threading.Event):
    # wait for 1st image
    while img_queue.empty():
        time.sleep(0.05)
    fps = 30
    img_size = img_queue.get(timeout=0.1).shape[:2][::-1]
    print('img_size = {}'.format(img_size))
    writer = cv2.VideoWriter('video.avi',
                             cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'),
                             fps,
                             img_size)
    while not stop_event.isSet() or not img_queue.empty():
        try:
            img = img_queue.get(timeout=0.1)
            writer.write(img)
        except queue.Empty:
            pass

    writer.release()
    print('Exiting: recording')


if __name__ == '__main__':
    print('*** Preview FLIR Camera ***')
    print('  q: quit')
    print('  h: help')
    print('  s: save')
    print('  r: record start/stop')

    record = False
    img_queue = queue.Queue()
    thread_record_exit = threading.Event()
    thread_record = threading.Thread(target=worker_record, args=(img_queue, thread_record_exit))
    thread_record.start()

    system = PySpin.System_GetInstance()

    cam_list = system.GetCameras()
    cam = cam_list.GetByIndex(0)
    cam.Init()

    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
    cam.GainAuto.SetValue(PySpin.GainAuto_Continuous)
    print('Acquisition mode set to continuous...')

    cam.BeginAcquisition()
    cam_serial_number = cam.DeviceSerialNumber.ToString()
    print('Camera {} start acquisition'.format(cam_serial_number))

    while True:
        img = cam.GetNextImage()
        if img.IsIncomplete():
            print('Image incomplete with image status %d ...' % img.GetImageStatus())
            continue

        width = img.GetWidth()
        height = img.GetHeight()
        img_converted = img.Convert(PySpin.PixelFormat_BGR8, PySpin.HQ_LINEAR)
        img_cv = img_converted.GetData().reshape((height, width, 3))
        if record:
            img_queue.put(img_cv)
            # img_queue.put(copy.copy(img_cv))
        img.Release()
        cv2.imshow('Preview', cv2.resize(img_cv, None, fx=0.6, fy=0.6))
        key = cv2.waitKey(5)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        elif key == ord('s'):
            cv2.imwrite('./tmp.png', img.GetData().reshape((img.GetHeight(), img.GetWidth())))
        elif key == ord('r'):
            # record
            record = not record
            if record:
                print('Recording')
            else:
                print('Stop recording')

    thread_record_exit.set()
    thread_record.join()
    cam.EndAcquisition()
    cam.DeInit()
    del cam
    cam_list.Clear()
    system.ReleaseInstance()
