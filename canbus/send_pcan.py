#!/usr/bin/env python3

"""
Shows how to use python to send a CAN msg using PCAN

1) pip install python-can
2) Install the PEAK can driver (tested on a Windows machine)
https://python-can.readthedocs.io/

Author: Zihan Chen
Date: Oct 23, 2021
"""

import can

# pcan: specifies the can bus type
# channel: PCAN-USB
# bitrate: 1Mbps
bus = can.interface.Bus(interface="pcan",
                        channel="PCAN_USBBUS1",
                        state=can.bus.BusState.ACTIVE,
                        bitrate=1000000)


# construct a CAN msg
msg = can.Message(arbitration_id=0x111,
                  data=[0x01, 0x02],
                  is_extended_id=False)

# send out 
bus.send(msg)
