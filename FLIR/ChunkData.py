#!/usr/bin/env python3

"""
Shows how to use ChunkData feature with QuickSpin API
"""

import PySpin


system = PySpin.System_GetInstance()

cam_list = system.GetCameras()
cam = cam_list.GetByIndex(0)
cam.Init()


# Activate Chunk mode
cam.ChunkModeActive.SetValue(True)

# ==========================================================
# ChunkSelector is used to select which ChunkData to access
# See ChunkSelectorEnums for a complete list and refer to
# the camera's technical reference manual for a list of supported
# ChunkData.
#
# Once ChunkData is selected, use ChunkEnable to set on/off
# ===========================================================
# print out current entry
print('ChunkSelect: {}'.format(cam.ChunkSelector.GetCurrentEntry().GetDisplayName()))

print('Selecting ExposureTime')
entry_exposure_time = cam.ChunkSelector.GetEntry(PySpin.ChunkSelector_ExposureTime)
cam.ChunkSelector.SetIntValue(entry_exposure_time.GetValue())
print('ChunkSelect: {}'.format(cam.ChunkSelector.GetCurrentEntry().GetDisplayName()))
print('ChunkSelect Value: {}'.format(cam.ChunkSelector.GetValue()))

entry_timestamp = cam.ChunkSelector.GetEntryByName('Timestamp')
cam.ChunkSelector.SetIntValue(entry_timestamp.GetValue())
print('ChunkSelect: {}'.format(cam.ChunkSelector.GetCurrentEntry().GetDisplayName()))
print('ChunkSelect Value: {}'.format(cam.ChunkSelector.GetValue()))

cam.DeInit()
del cam
cam_list.Clear()
system.ReleaseInstance()
