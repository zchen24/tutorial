#!/usr/bin/env python3

"""
Example: toggle an LED
"""

import pysoem
import time


VENDOR_ID = 0x34E
PRODUCT_CODE = 0xBB

master = pysoem.Master()
master.open('eth0')
num_slaves = master.config_init()
if num_slaves <= 0:
    print('No slave is connected, exiting')
    master.close()
    exit(-1)

slave = master.slaves[0]
if (slave.man != VENDOR_ID) or (slave.id != PRODUCT_CODE):
    print('Unexpected slave layout, exiting')
    master.close()
    exit(-2)

# config io map
master.config_map()

# check state (SAFEOP)
if master.state_check(pysoem.SAFEOP_STATE, 50000) != pysoem.SAFEOP_STATE:
    print('not all slaves reached SAFEOP state, state = {}'.format(master.state))
    master.close()
    exit(-2)

master.state = pysoem.OP_STATE
master.write_state()
for i in range(40):
    master.state_check(pysoem.OP_STATE, 50000)
    if master.state == pysoem.OP_STATE:
        print('all slaves reached OP_STATE')
        break

for i in range(10000):
    master.send_processdata()
    actual_wkc = master.receive_processdata(10000)
    if actual_wkc != master.expected_wkc:
        print('incorrect wkc')

    # pdo update loop
    tmp = bytearray(slave.output)
    # toggle LED2 every 0.5 sec
    if i % 50 == 0:
        tmp[0] ^= 0b11111100
        # print('input buttons = {}'.format(slave.input[0]))
    tmp[0] = (tmp[0] & 0b11111100) + (slave.input[0] ^ 0b11)
    slave.output = bytes(tmp)
    time.sleep(0.01)

master.state = pysoem.INIT_STATE
master.write_state()
master.close()
