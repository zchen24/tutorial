#!/usr/bin/env python3

"""
Multicast (UDP) receive example.

Reference:
[1] https://pymotw.com/2/socket/multicast.html
[2] https://www.tldp.org/HOWTO/text/Multicast-HOWTO
"""


import socket
import struct


multicast_group = '224.3.29.71'
multicast_port = 10000
server_address = ('', multicast_port)


sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
# Bind to the server address (local IP (any ip), port is 10000)
sock.bind(server_address)

# Kernel add the socket to the multicast group on all interfaces
group = socket.inet_aton(multicast_group)
mreq = struct.pack('4sL', group, socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

while True:
    print('\nwaiting to receive message')
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes

    print("received {} bytes from {}:".format(len(data), addr))
    print("{}".format(data.decode('utf-8')))

    print('sending acknowledge to {}'.format(addr))
    sock.sendto('ack'.encode('utf-8'), addr)
