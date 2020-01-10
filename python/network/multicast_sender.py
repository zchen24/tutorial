#!/usr/bin/env python3

"""
Multicast (UDP) receive example.

Reference:
[1] https://pymotw.com/2/socket/multicast.html
[2] https://www.tldp.org/HOWTO/text/Multicast-HOWTO
"""

import socket
import struct


message = 'very important data'
multicast_group = '224.3.29.71'
multicast_port = 10000
server_address = ('', multicast_port)


sock = socket.socket(socket.AF_INET,     # Internet
                     socket.SOCK_DGRAM)  # UDP

# Bind to the server address
sock.settimeout(0.2)
# Set the time-to-live for msg to 1 so they do not go past the local network
ttl = struct.pack('b', 1)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)

try:
    # send data to the multicast group
    print('sending msg: {}'.format(message))
    sent = sock.sendto(message.encode('utf-8'), (multicast_group, multicast_port))
    # look for responses from all recipients
    while True:
        print('waiting to receive')
        try:
            data, server = sock.recvfrom(16)
        except socket.timeout:
            print('timed out, no more responses')
            break
        else:
            print('received {} from {}'.format(data.decode('utf-8'), server))
finally:
    sock.close()
