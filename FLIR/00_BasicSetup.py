#!/usr/bin/env python3

"""
Shows how to use ChunkData feature with QuickSpin API
"""

import PySpin


system = PySpin.System_GetInstance()

cam_list = system.GetCameras()

# Or by serial number
# cam_sn = 12345
# cam = cam_list.GetBySerial(cam_sn)
cam = cam_list.GetByIndex(0)
cam.Init()

print('Fancy processing')

cam.DeInit()
del cam
cam_list.Clear()
system.ReleaseInstance()
