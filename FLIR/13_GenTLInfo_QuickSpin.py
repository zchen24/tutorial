#!/usr/bin/env python3

"""
Querying GenICam TL (Transport Layer) info using QuickSpin API.
Based on GenTLInfo_QuickSpin.cpp example.

Date: 2019-12-22
"""

import PySpin

def print_version(system):
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

def PrintTransportLayerInterfaceInfo(iface):
    try:
        print('Interface display name: {}'.format(iface.TLInterface.InterfaceDisplayName.ToString()))
        print('Interface ID : {}'.format(iface.TLInterface.InterfaceID.ToString()))
        print('Interface type : {}\n'.format(iface.TLInterface.InterfaceType.ToString()))
    except PySpin.SpinnakerException as e:
        print("Error: {}".format(e))


def PrintTransportLayerDeviceInfo(cam):
    try:
        print("Device serial number: {}".format(cam.TLDevice.DeviceSerialNumber.ToString()))
        print("Device vendor number: {}".format(cam.TLDevice.DeviceVendorName.ToString()))
        print("Device display number: {}\n".format(cam.TLDevice.DeviceDisplayName.ToString()))
    except PySpin.SpinnakerException as e:
        print("Error: {}".format(e))


def PrintTransportLayerStreamInfo(cam):
    try:
        print("Stream ID: {}".format(cam.TLStream.StreamID.ToString()))
        print("Stream type: {}".format(cam.TLStream.StreamType.ToString()))
    except PySpin.SpinnakerException as e:
        print("Error: {}".format(e))


def PrintApplicationLayerDeviceInfo(cam):
    try:
        print("Exposure time: {}".format(cam.ExposureTime.ToString()))
        print("Black level: {}".format(cam.BlackLevel.ToString()))
        print("Height: {}".format(cam.Height.ToString()))
    except PySpin.SpinnakerException as e:
        print("Error: {}".format(e))


if __name__ == '__main__':
    system = PySpin.System_GetInstance()
    print_version(system)

    cam_list = system.GetCameras()
    print("Number of cameras detected: {}".format(len(cam_list)))

    if_list = system.GetInterfaces()
    print("Number of interfaces detected: {}".format(len(if_list)))

    print("*** PRINTING INTERFACE INFORMATION ***\n")
    for iface in if_list:
        PrintTransportLayerInterfaceInfo(iface)

    print("*** PRINTING TRANSPORT LAYER DEVICE INFORMATION ***\n")
    for cam in cam_list:
        cam.Init()
        PrintTransportLayerDeviceInfo(cam)
        cam.DeInit()

    print("*** PRINTING TRANSPORT LAYER STREAMING INFORMATION ***\n")
    for cam in cam_list:
        cam.Init()
        PrintTransportLayerStreamInfo(cam)
        cam.DeInit()

    print("*** PRINTING APPLICATION LAYER INFORMATION ***\n")
    for cam in cam_list:
        cam.Init()
        PrintApplicationLayerDeviceInfo(cam)
        cam.DeInit()

    cam_list.Clear()
    if_list.Clear()
    system.ReleaseInstance()
