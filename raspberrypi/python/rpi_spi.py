#!/usr/bin/env python3

"""
Show how to use SPI bus


"""

import spidev


spi_bus = 0     # spi bus number
spi_dev = 1     # spi chip_select pin 0 or 1


spi = spidev.SpiDev()
spi.open(spi_bus, spi_dev)

spi.max_speed_hz = 500000
spi.mode = 0

msg = [0x76]

# TO FINISH

spi.close()