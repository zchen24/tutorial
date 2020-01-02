#!/usr/bin/env python3

"""
Shows how to use Sequencer feature:
We set up a sequence of configurations for the camera (e.g. exposure time, gain).
When the camera starts capturing, it loops through the list of configs. In this
example, there are 5 different configs in the sequence and we retrieve 10 images
(i.e. 2 rounds). The active sequencer set can be retrieved via ChunkData.

The 10 images have the following configs:
00  sequencer set 0
01  sequencer set 1
02  sequencer set 2
03  sequencer set 3
04  sequencer set 4
05  sequencer set 0
06  sequencer set 1
07  sequencer set 2
08  sequencer set 3
09  sequencer set 4

Date: 2019-12-31
"""

import PySpin


def set_single_state(cam: PySpin.CameraPtr, sequence_number, exposure_time_to_set, gain_to_set):
    cam.SequencerSetSelector.SetValue(sequence_number)
    print('Setting state {}...'.format(sequence_number))

    if exposure_time_to_set > cam.ExposureTime.GetMax():
        exposure_time_to_set = cam.ExposureTime.GetMax()
    cam.ExposureTime.SetValue(exposure_time_to_set)
    print('\tExposure set to {0:.0f}...'.format(cam.ExposureTime.GetValue()))

    if gain_to_set > cam.Gain.GetMax():
        gain_to_set = cam.Gain.GetMax()
    cam.Gain.SetValue(gain_to_set)
    print('\tGain set to {0:.5f}...'.format(cam.Gain.GetValue()))

    cam.SequencerTriggerSource.SetValue(PySpin.SequencerTriggerSource_FrameStart)
    print('\tTrigger source set to start of frame...')

    final_sequence_index = 4
    if sequence_number == final_sequence_index:
        cam.SequencerSetNext.SetValue(0)
    else:
        cam.SequencerSetNext.SetValue(sequence_number + 1)
    print('\tNext state set to {}...'.format(cam.SequencerSetNext.GetValue()))

    cam.SequencerSetSave.Execute()
    print('Current state saved...\n')


system = PySpin.System_GetInstance()

cam_list = system.GetCameras()
cam = cam_list.GetByIndex(0)
cam.Init()


# -----------------------------------
# configure Sequencer
# -----------------------------------
cam.ChunkModeActive.SetValue(True)
cam.ChunkSelector.SetValue(PySpin.ChunkSelector_SequencerSetActive)
cam.ChunkEnable.SetValue(True)

# turn off SequencerMode before configuration
# if current Sequencer is valid, then manually turn off SequencerMode
# else, it isn't, then we know that the SequencerMode is already off
if cam.SequencerConfigurationValid.GetValue() == PySpin.SequencerConfigurationValid_Yes:
    cam.SequencerMode.SetValue(PySpin.SequencerMode_Off)
print('Sequencer mode disabled...')

# turn off auto exposure and auto gain to allow manual control
cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
cam.GainAuto.SetValue(PySpin.GainAuto_Off)
# turn on Sequencer config mode
cam.SequencerConfigurationMode.SetValue(PySpin.SequencerConfigurationMode_On)


# -----------------------------------
# configure individual states
# -----------------------------------
exposure_time_to_set = cam.ExposureTime.GetMin()
gain_to_set = cam.Gain.GetMin()

exposure_time_max_to_set = 2000000
gain_max = cam.Gain.GetMax()
for sequence_number in range(5):
    print('exposure_time_to_set = {}  gain_to_set = {}'.format(exposure_time_to_set, gain_to_set))
    set_single_state(cam,
                     sequence_number,
                     exposure_time_to_set,
                     gain_to_set
                     )
    # increment value
    exposure_time_to_set += exposure_time_max_to_set / 10.0
    gain_to_set += gain_max / 50.0


# -----------------------------------
# configure Sequencer 2
# -----------------------------------
cam.SequencerConfigurationMode.SetValue(PySpin.SequencerConfigurationMode_Off)
print('Sequencer configuration mode disabled...')
cam.SequencerMode.SetValue(PySpin.SequencerMode_On)
print('Sequencer mode enabled...')
# validate sequencer settings

if cam.SequencerConfigurationValid.GetValue() == PySpin.SequencerConfigurationValid_Yes:
    print('Sequencer configuration valid...\n')
else:
    print('Sequencer configuration not valid. Aborting...\n')
    exit(-1)


print('*** IMAGE ACQUISITION ***\n')
cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
print('Acquisition mode set to continuous...')

cam.BeginAcquisition()
print('Acquiring images...')

NUM_IMAGES = 10
serial_number = cam.DeviceSerialNumber.GetValue()
for i in range(NUM_IMAGES):
    img = cam.GetNextImage()
    print('Grabbed image {}, width = {}, height = {}, sequencer set = {}'.format(
        i, img.GetWidth(), img.GetHeight(), img.GetChunkData().GetSequencerSetActive()))
    filename = 'Sequencer-{}-{}.jpg'.format(serial_number, i)
    img.Save(filename)
    print('Saving {}'.format(filename))
    img.Release()

cam.EndAcquisition()


# Reset sequencer
cam.SequencerMode.SetValue(PySpin.SequencerMode_Off)
print('Turning off sequencer mode...')
cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
print('Turning automatic exposure back on...')
cam.GainAuto.SetValue(PySpin.GainAuto_Continuous)
print('Turning automatic gain mode back on...\n')
cam.ChunkModeActive.SetValue(True)
print('Turning ChunkData mode off...\n')


cam.DeInit()
del cam
cam_list.Clear()
system.ReleaseInstance()
