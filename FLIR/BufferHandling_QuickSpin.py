#!/usr/bin/env python3

"""
FLIR Buffer Handling Mode using QuickSpin API

Date: 2019-12-22

Reference:
1. Understanding Buffer Handling Applicable products: http://tinyurl.com/vthoq8f
2. Based on BufferHandling.py example.

Note on Newest First mode
Captured ID: 0 => 1 => 2 => 3 => 4 => 5

 Host buffer: 0, 1, 2
 Host read Frame 2
 Cam transfers Frame 3 as there is one buffer space available
 Host buffer: 0, 1, 3
 Host read Frame 3
 Cam transfers Frame 4 as there is one buffer space available
 Host buffer: 0, 1, 4
 Host read Frame 4

 => Host reads Frame 2, 3, 4

 If we continue to read
 Cam transfers Frame 5 as there is one buffer space available
 Host buffer: 0, 1, 5
 Host read Frame 5
 Host buffer: 0, 1
 Host read Frame 1
 Host buffer: 0
 Host read Frame 0
 Host buffer: empty!

 => Host reads Frame 2, 3, 4, 5, 1, 0
"""


import PySpin
import time


NUM_BUFFERS = 3    # Number of buffers (on host side)
NUM_TRIGGERS = 6   # Number of triggers
NUM_TRANSFERS = 6  # Number of transfers


def print_version(system):
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))


system = PySpin.System_GetInstance()
print_version(system)

cam_list = system.GetCameras()
cam = cam_list.GetByIndex(0)
cam.Init()

# Set Trigger Mode
# The trigger must be disabled in order to configure the trigger source
print('===== Configuring trigger: software =====')
cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
cam.TriggerMode.SetValue(PySpin.TriggerMode_On)

print('===== Acquiring images =====')

cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
print('Acquisition mode set to continuous...')

device_serial_number = cam.DeviceSerialNumber.ToString()
print('Device serial number retrieved as {}...'.format(device_serial_number))

cam.TLStream.StreamBufferCountMode.SetValue(PySpin.StreamBufferCountMode_Manual)
print('Stream Buffer Count Mode set to manual...')

buffer_count = cam.TLStream.StreamBufferCountManual
# Display Buffer Info
print('\nDefault Buffer Handling Mode: %s' % cam.TLStream.StreamBufferHandlingMode.GetCurrentEntry().GetDisplayName())
print('Default Buffer Count: %d' % buffer_count.GetValue())
print('Maximum Buffer Count: %d' % buffer_count.GetMax())

buffer_count.SetValue(NUM_BUFFERS)
print('Buffer count now set to: %d' % buffer_count.GetValue())
print('\nCamera will be triggered %d times in a row before %d images will be retrieved' % (
NUM_TRIGGERS, NUM_TRANSFERS))

cam.TLStream.StreamBufferHandlingMode.SetValue(PySpin.StreamBufferHandlingMode_NewestFirst)
print('\n\nBuffer Handling Mode has been set to %s' % cam.TLStream.StreamBufferHandlingMode.GetCurrentEntry().GetDisplayName())

cam.BeginAcquisition()
time.sleep(1)

# Grab 6 images by trigger
for cnt in range(NUM_TRIGGERS):
    cam.TriggerSoftware.Execute()
    print('Camera triggered {}. No image grabbed'.format(cnt))
    # sleep 0.25 sec: allow trigger to finish (e.g. exposure, transfer to host)
    time.sleep(0.25)

try:
    for cnt in range(NUM_TRANSFERS):
        grab_timeout = 500
        img = cam.GetNextImage(grab_timeout)
        if img.IsIncomplete():
            print('Image incomplete with image status %s ...\n' % img.GetImageStatus())
        else:
            print('Frame ID: {}'.format(img.GetFrameID()))
            img.Save('{}-{}-{}-{}.jpg'.format(cam.TLStream.StreamBufferHandlingMode.GetCurrentEntry().GetSymbolic(),
                                              cam.DeviceSerialNumber.ToString(),
                                              cnt,
                                              img.GetFrameID()))
        img.Release()
        time.sleep(0.25)
except PySpin.SpinnakerException as e:
    print('Error: {}'.format(e))
    if cam.TLStream.StreamBufferHandlingMode.GetCurrentEntry().GetSymbolic() == 'NewestOnly':
        print('Error should occur when grabbing image 1 with handling mode set to NewestOnly')

cam.EndAcquisition()

print('Resetting trigger')
cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)

del cam
cam_list.Clear()
system.ReleaseInstance()
