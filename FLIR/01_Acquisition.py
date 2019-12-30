#!/usr/bin/env python3

"""
Shows how to acquire an image. See Acquisition.py example
Date: 2019-12-29
"""

import PySpin


system = PySpin.System_GetInstance()

cam_list = system.GetCameras()
cam = cam_list.GetByIndex(0)
cam.Init()

print('*** IMAGE ACQUISITION ***\n')
cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
print('Acquisition mode set to continuous...')

cam.BeginAcquisition()
cam_serial_number = cam.DeviceSerialNumber.ToString()
NUM_IMAGES = 10
for i in range(NUM_IMAGES):
    img = cam.GetNextImage()
    if img.IsIncomplete():
        print('Image incomplete with image status %d ...' % img.GetImageStatus())
    else:
        width = img.GetWidth()
        height = img.GetHeight()
        print('Grabbed Image %d, width = %d, height = %d' % (i, width, height))
        img_converted = img.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
        filename = 'Acquisition-{}-{:02}.jpg'.format(cam_serial_number, i)
        img_converted.Save(filename)
        print('Image saved at {}'.format(filename))
        # release in order to keep from filling the buffer
        img.Release()
cam.EndAcquisition()

cam.DeInit()
del cam
cam_list.Clear()
system.ReleaseInstance()
