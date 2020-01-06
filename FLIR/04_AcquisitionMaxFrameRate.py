#!/usr/bin/env python3

"""
Shows how to stream video max speed (over 200 FPS)
Author: Zihan Chen
Date: 2020-01-06
"""

import PySpin


system = PySpin.System_GetInstance()

cam_list = system.GetCameras()
cam = cam_list.GetByIndex(0)
cam.Init()

cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
print('Acquisition mode set to continuous...')

# Turn off all auto algorithm: Exposure/Gain/WhiteBalance
cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
cam.ExposureTime.SetValue(2000)
cam.GainAuto.SetValue(PySpin.GainAuto_Off)
cam.Gain.SetValue(1.0)
cam.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Off)

# Set AcquisitionFrameRate
cam.AcquisitionFrameRateEnable.SetValue(True)
cam.AcquisitionFrameRate.SetValue(226)
print('Result frame rate = {} fps\n'.format(cam.AcquisitionResultingFrameRate.GetValue()))

# Start acquisition
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

cam.DeInit()
del cam
cam_list.Clear()
system.ReleaseInstance()
