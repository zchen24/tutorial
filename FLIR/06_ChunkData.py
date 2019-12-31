#!/usr/bin/env python3

"""
Shows how to use ChunkData feature with QuickSpin API
This example enables ChunkData mode with entries:
1. ExposureTime
2. FrameID

"""

import time
import PySpin


system = PySpin.System_GetInstance()

cam_list = system.GetCameras()
cam = cam_list.GetByIndex(0)
cam.Init()


# ==========================================================
# ChunkSelector is used to select which ChunkData to access
# See ChunkSelectorEnums for a complete list and refer to
# the camera's technical reference manual for a list of supported
# ChunkData.
#
# Once ChunkData is selected, use ChunkEnable to set on/off
# ===========================================================

# activate Chunk mode
cam.ChunkModeActive.SetValue(True)

# enable ExposureTime & FrameID
cam.ChunkSelector.SetValue(PySpin.ChunkSelector_ExposureTime)
print('ChunkSelect: {}'.format(cam.ChunkSelector.GetCurrentEntry().GetDisplayName()))
cam.ChunkEnable.SetValue(True)
print('ChunkData: ExposureTime enabled')

cam.ChunkSelector.SetValue(PySpin.ChunkSelector_FrameID)
cam.ChunkEnable.SetValue(True)

# continuous acquisition
cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

cam.TLStream.StreamBufferHandlingMode.SetValue(PySpin.StreamBufferHandlingMode_NewestOnly)
cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)


cam.BeginAcquisition()
exposures = [1000, 2000, 3000, 4000, 5000]
for exp in exposures:
    cam.ExposureTime.SetValue(exp)
    time.sleep(0.05)
    img = cam.GetNextImage()
    chunk_data = img.GetChunkData()
    print('FrameID = {} \t Exposure = {}'.format(chunk_data.GetFrameID(),
                                                 chunk_data.GetExposureTime()))
    img.Release()

cam.EndAcquisition()

# Deactivate ChunkData mode
cam.ChunkModeActive.SetValue(False)

cam.DeInit()
del cam
cam_list.Clear()
system.ReleaseInstance()
