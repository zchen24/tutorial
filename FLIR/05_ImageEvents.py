#!/usr/bin/env python3

"""
Shows how to use Events

Author: Zihan Chen
Date: 2019-12-30
"""

import PySpin


class InterfaceEventHandler(PySpin.InterfaceEvent):
    def __init__(self):
        super(InterfaceEventHandler, self).__init__()

    def OnDeviceArrival(self, serialNumber):
        print('A camera arrived: SN {}'.format(serialNumber))

    def OnDeviceRemoval(self, serialNumber):
        print('A camera was removed: SN {}'.format(serialNumber))

system = PySpin.System_GetInstance()
handler = InterfaceEventHandler()
system.RegisterInterfaceEvent(handler)
input('Press enter to quit\n')
system.ReleaseInstance()
