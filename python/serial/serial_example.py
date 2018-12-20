#!/usr/bin/env python

# sample code for using Serial port
# 2013-07-15
# Zihan Chen

# pyserial documentation
# http://pyserial.sourceforge.net/

from __future__ import print_function
import serial


if __name__ == "__main__":
    # serial port
    ser = serial.Serial('/dev/ttyUSB0', 115200)
    # ser.open()
    if ser.isOpen():
        print("serial is open")
        data = ser.read(100)
        print(data)
        pass

    # send byte(s)
    # this send byte 15
    ser.write(serial.to_bytes([15]))
