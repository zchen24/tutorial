#!/usr/bin/env python3

"""
Example Modbus Server: run 60 seconds
Reference: https://youtu.be/FYPQgnQE9fk

Author: Zihan
Date: 2021-07-10
"""

from pyModbusTCP.server import ModbusServer, DataBank
import time

try:
    server = ModbusServer(host='localhost',
                          port=1234,
                          no_block=True)
    server.start()
    state = 10
    t0 = time.time()
    while True:
        DataBank.set_words(0, [1, 2, 3])
        if state != DataBank.get_words(10, 1)[0]:
            state = DataBank.get_words(10, 1)[0]
            print('state changed to {}'.format(state))
        time.sleep(0.05)
        if time.time() - t0 > 60:
            break
    server.stop()
except:
    print('Failed to create a server')
