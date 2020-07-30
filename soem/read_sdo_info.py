#!/usr/bin/env python

"""Read SDO Info + Read/Write SDO"""

import sys
import pysoem


def read_sdo_info(ifname):
    master = pysoem.Master()
    
    master.open(ifname)
    
    if master.config_init() > 0:
    
        for slave in master.slaves:
            try:
                od = slave.od
            except pysoem.SdoInfoError:
                print('no SDO info for {}'.format(slave.name))
            else:
                print(slave.name)

                for obj in od:
                    print(' Idx: {}; Code: {}; Type: {}; BitSize: {}; Access: {}; Name: "{}"'.format(
                        hex(obj.index),
                        obj.object_code,
                        obj.data_type,
                        obj.bit_length,
                        hex(obj.obj_access),
                        obj.name))
                    for i, entry in enumerate(obj.entries):
                        if entry.data_type > 0 and entry.bit_length > 0:
                            print('  Subindex {}; Type: {}; BitSize: {}; Access: {} Name: "{}"'.format(
                                i,
                                entry.data_type,
                                entry.bit_length,
                                hex(entry.obj_access),
                                entry.name))

    else:
        print('no slave available')
        
    master.close()


if __name__ == '__main__':
    ifname = 'eth0'
    master = pysoem.Master()
    master.open(ifname)
    master.config_init()
    slave = master.slaves[0]

    # Index, SubIndex
    slave.sdo_read(0x1001, 0)
