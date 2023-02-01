#!/usr/bin/env python3

"""
Example script that demos how to parse a PCAN trace file (trc)

Reference: PEAK CAN TRC File Format
https://www.peak-system.com/produktcd/Pdf/English/PEAK_CAN_TRC_File_Format.pdf

Author: Zihan Chen
Date: 2023-02-02
"""


import logging
import time
import matplotlib.pyplot as plt

class CanMsg:
    def __int__(self, txt=None):
        self.msg_number = 0
        self.t = 0
        self.type = 'DT'
        self.id = 0x00000000
        self.d_len = 0
        self.d_bits = 0
        self.rx = True
        self.data = []
        if txt is not None:
            self.parse(txt)

    def parse(self, txt:str, type='peak'):
        if type != 'peak':
            logging.error("Invalid logging type")
            return

        data = txt.split()
        if len(data) < 7:
            logging.error('Parse error')
        elif (int(data[5]) + 6) != len(data):
            logging.error('Data length error')
        else:
            self.msg_number = int(data[0])
            self.t = float(data[1])
            self.type = data[2]
            self.id = data[3]
            self.rx = True
            self.d_len = int(data[5])
            self.data = data[6:]
            self.d_bits = (5 + self.d_len) * 8

    @property
    def src_id(self):
        return int(self.id[2:4], 16)

    @property
    def dst_id(self):
        return int(self.id[4:6], 16)

    @property
    def cmd_id(self):
        return int(self.id[6:], 16)


def find_msg(msgs: list[CanMsg], src_id, dst_id, cmd_id):
    msgs_found = []
    for m in msgs:
        if m.src_id == src_id and m.dst_id == dst_id and m.cmd_id == cmd_id:
            msgs_found.append(m)
    return msgs_found


if __name__ == '__main__':
    f_name = "example.trc"
    f = open(f_name)
    lines = f.readlines()[17:]
    f.close()

    # read all msgs
    all_msgs = []
    for l in lines:
        msg = CanMsg()
        msg.parse(l)
        all_msgs.append(msg)

    # compute bus load
    # dt: ms
    dt = all_msgs[-1].t - all_msgs[0].t

    # compute all data payload
    sum_bits = 0
    for m in all_msgs:
        sum_bits += m.d_bits

    BAUD_RATE = 500000
    avg_load = sum_bits / dt * 1000.0
    avg_load_percent = avg_load / BAUD_RATE
    print("Average CAN bus load = {:.2f}%".format(avg_load_percent * 100))

    a_msgs = find_msg(all_msgs, 0x78, 0x82, 0x45)
    print("Found {} of a_msgs".format(len(a_msgs)))