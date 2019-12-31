#!/usr/bin/env python3

"""
Shows how to use the Trigger feature:
The camera only take an image when the trigger event happens.
When using SoftwareTrigger, SoftwareTrigger needs to be called.
Blackfly camera also support other triggers e.g. GPIO Input.

Reference: Acquisition Control section in Technical Reference.

Date: 2019-12-31
"""

import PySpin


system = PySpin.System_GetInstance()

cam_list = system.GetCameras()
cam = cam_list.GetByIndex(0)
cam.Init()


# turn off trigger mode before configuration
cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

cam.BeginAcquisition()
try:
    img = cam.GetNextImage(1000)
    img.Release()
except PySpin.SpinnakerException as e:
    print('Expected timeout, no image taken yet: {}'.format(e))

cam.TriggerSoftware.Execute()
img = cam.GetNextImage(1000)
print('One image taken FrameID: {}'.format(img.GetFrameID()))
img.Release()

cam.EndAcquisition()

# reset trigger mode to off
cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)

cam.DeInit()
del cam
cam_list.Clear()
system.ReleaseInstance()
