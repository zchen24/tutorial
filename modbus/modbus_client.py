#!/usr/bin/env python3

"""
Example Modbus client:
Reference: https://youtu.be/FYPQgnQE9fk

Author: Zihan
Date: 2021-07-10
"""

from pyModbusTCP.client import ModbusClient

client = ModbusClient(host='localhost', port=1234)
client.open()

print('reg = {}'.format(client.read_holding_registers(0, 3)))
client.write_single_register(10, 15)
client.close()