#!/usr/bin/env python3

"""
Shows how to convert between int and bytes.
Use case:
Decode data to/from hardware e.g. Serial, SPI, I2C
"""

if __name__ == '__main__':
    data_bytes = [0x01, 0x02]
    int_big_signed = int.from_bytes(data_bytes, byteorder='big', signed=True)
    int_little_signed = int.from_bytes(data_bytes, byteorder='little', signed=True)
    print('int_big_signed = {}'.format(int_big_signed))
    print('int_little_signed = {}'.format(int_little_signed))

    a_int = 258
    bytes_big_signed = a_int.to_bytes(2, byteorder='big', signed=True)
    bytes_little_signed = a_int.to_bytes(2, byteorder='little', signed=True)
    print('bytes_big_signed = {}'.format(bytes_big_signed))
    print('bytes_little_signed = {}'.format(bytes_little_signed))
