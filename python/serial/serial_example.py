#!/usr/bin/env python

# sample code for using Serial port
# 2013-07-15
# Zihan Chen

# pyserial documentation
# http://pyserial.sourceforge.net/

# pyserial
import serial

def open_serial():
    # serial port
    ser = serial.Serial('/dev/ttyUSB0', 115200)
    
    ser.open()
    if ser.isOpen():
        print "serial is open"
        data = ser.read(100)
        print data
        pass

if __name__ == "__main__":
    open_serial()

