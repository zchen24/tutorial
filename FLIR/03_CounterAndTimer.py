#!/usr/bin/env python3

"""
Shows how to use FLIR camera's timer
See Exposure_QuickSpin.py example

This program sets camera to different exposure time,

Please see Counter And Timer Control section in camera's
technical reference.

Author: Zihan Chen
Date: 2019-12-30
"""

import numpy as np
import PySpin


def setup_counter_and_timer(cam: PySpin.CameraPtr):
    """
    Setup a PWM using Counter & Timer. The signal is
    set to run at 50 Hz and 70% duty cycle.

    CounterEventSource: event to increment the counter
    CounterTriggerSource: event to start the counter
    """
    if not PySpin.IsAvailable(cam.CounterSelector):
        print('\nCamera does not support Counter and Timer Functionality.  Aborting...\n')
        return False

    cam.CounterSelector.SetIntValue(PySpin.CounterSelector_Counter0)
    cam.CounterEventSource.SetValue(PySpin.CounterEventSource_MHzTick)
    cam.CounterDuration.SetValue(14000) # 14 ms
    cam.CounterDelay.SetValue(6000)     # 6 ms
    duty_cycle = cam.CounterDuration.GetValue() / \
                 (cam.CounterDuration.GetValue() + cam.CounterDelay.GetValue()) * 100
    pulse_rate = 1000000 / float(cam.CounterDuration.GetValue() + cam.CounterDelay.GetValue())
    print('\nThe duty cycle has been set to {}%'.format(duty_cycle))
    print('\nThe pulse rate has been set to {} Hz'.format(pulse_rate))

    cam.CounterTriggerSource.SetValue(PySpin.CounterTriggerSource_FrameTriggerWait)
    cam.CounterTriggerActivation.SetValue(PySpin.CounterTriggerActivation_LevelHigh)


def configure_digital_io(cam: PySpin.CameraPtr):
    """
    See Digital IO Control section
    For Blackfly Line1 is Opto-isolated output
                 Line2 is 3.3V output
    """
    cam.LineSelector.SetValue(PySpin.LineSelector_Line1)
    cam.LineMode.SetValue(PySpin.LineMode_Output)
    cam.LineSource.SetValue(PySpin.LineSource_Counter0Active)
    cam.LineSelector.SetValue(PySpin.LineSelector_Line2)
    cam.V3_3Enable.SetValue(True)


def configure_exposure_and_trigger(cam: PySpin.CameraPtr):
    """
    Set exposure to manual mode, 5 ms
    Set trigger to ON and source to Counter0Start
    Set trigger overlay to READOUT
    """
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    cam.ExposureTime.SetValue(5000) # 5 ms

    cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
    cam.TriggerSource.SetValue(PySpin.TriggerSource_Counter0Start)
    cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
    cam.TriggerMode.SetValue(PySpin.TriggerMode_On)


def reset_trigger(cam: PySpin.CameraPtr):
    cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)


if __name__ == '__main__':
    system = PySpin.System_GetInstance()
    version = system.GetLibraryVersion()
    print('Library version: {}.{}.{}.{}'.format(version.major, version.minor, version.type, version.build))

    cam_list = system.GetCameras()
    cam = cam_list.GetByIndex(0)
    cam.Init()

    setup_counter_and_timer(cam)
    configure_exposure_and_trigger(cam)
    cam.BeginAcquisition()

    NUM_IMAGES = 10
    timestamps = []
    for i in range(NUM_IMAGES):
        img = cam.GetNextImage()
        timestamps.append(img.GetTimeStamp())
        img.Save('Timer_{:03}.jpg'.format(i))
        img.Release()
    cam.EndAcquisition()
    timestamps = np.array(timestamps)
    print((timestamps - timestamps[0])/1000000)

    cam.DeInit()
    del cam
    cam_list.Clear()
    system.ReleaseInstance()
