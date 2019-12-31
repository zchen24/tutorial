#!/usr/bin/env python3

"""
Shows how to save image streams as AVI files.
Comment: this feature is handy, but using OpenCV to save is preferred

Date: 2019-12-29
"""

import PySpin


def save_to_avi(images):
    """
    Save to AVI using PySpin
    """
    print('Saving to Uncompressed...')
    avi_recorder = PySpin.SpinVideo()
    option = PySpin.AVIOption()
    option.frameRate = cam.AcquisitionFrameRate.GetValue()
    filename = 'SaveToAvi-Uncompressed'
    avi_recorder.Open(filename, option)
    for i, img in enumerate(images):
        avi_recorder.Append(img)
        print('Appended image {}...'.format(i))
    avi_recorder.Close()

    print('Saving to H264...')
    option = PySpin.H264Option()
    option.frameRate = cam.AcquisitionFrameRate.GetValue()
    option.bitrate = 1000000
    option.height = images[0].GetHeight()
    option.width = images[0].GetWidth()
    filename = 'SaveToAvi-H264'
    avi_recorder.Open(filename, option)
    for i, img in enumerate(images):
        avi_recorder.Append(img)
    avi_recorder.Close()


system = PySpin.System_GetInstance()

cam_list = system.GetCameras()
cam = cam_list.GetByIndex(0)
cam.Init()

images = []
cam.BeginAcquisition()
for i in range(200):
    img = cam.GetNextImage()
    images.append(img.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR))
    img.Release()
cam.EndAcquisition()

save_to_avi(images)

cam.DeInit()
del cam
cam_list.Clear()
system.ReleaseInstance()
