#!/usr/bin/env python3

"""
Shows how to set Acquisition frame rate manually
Note: exposure time can limit the maximum frame rate

Date: 2019-12-31
"""

import PySpin


system = PySpin.System_GetInstance()

cam_list = system.GetCameras()
cam = cam_list.GetByIndex(0)
cam.Init()

cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
print('Acquisition mode set to continuous...')

# ----------------------------------------
# ExposureTime can affect max FrameRate
# ----------------------------------------
cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
print('Exposure auto mode set to off...')
cam.ExposureTime.SetValue(10000) # 10 ms
print('\n\nSetting exposure time to 10 ms')

cam.AcquisitionFrameRateEnable.SetValue(True)
print('Enabling AcquisitionFrameRate mode to allow manually set FrameRate')
cam.AcquisitionFrameRate.SetValue(200)
print('Setting acquisition frame rate to 200')
print('Result frame rate = {} fps\n'.format(cam.AcquisitionResultingFrameRate.GetValue()))

cam.BeginAcquisition()
NUM_IMAGES = 10
t_start = None
for i in range(NUM_IMAGES):
    img = cam.GetNextImage()
    if t_start is None:
        t_start = img.GetTimeStamp()
    print('FrameID = {}, Timestamp = {:.03f} ms'.format(img.GetFrameID(),
                                                   (img.GetTimeStamp() - t_start)/1000000))
    img.Release()
cam.EndAcquisition()


print('Setting acquisition frame rate to 200')
cam.ExposureTime.SetValue(2000) # 2 ms
cam.AcquisitionFrameRate.SetValue(200)
print('Result frame rate = {} fps\n'.format(cam.AcquisitionResultingFrameRate.GetValue()))

cam.BeginAcquisition()
NUM_IMAGES = 10
t_start = None
for i in range(NUM_IMAGES):
    img = cam.GetNextImage()
    if t_start is None:
        t_start = img.GetTimeStamp()
    print('FrameID = {}, Timestamp = {:.03f} ms'.format(img.GetFrameID(),
                                                   (img.GetTimeStamp() - t_start)/1000000))
    img.Release()
cam.EndAcquisition()

# reset
cam.AcquisitionFrameRateEnable.SetValue(False)
cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)

cam.DeInit()
del cam
cam_list.Clear()
system.ReleaseInstance()
