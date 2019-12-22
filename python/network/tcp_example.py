#!/usr/bin/env python
""" TCP Example
Example:
    # start server
    $ tcp_example.py -s
    > received data: hello

    # start client
    $ tcp_example.py -c hello
    > Sent    : hello
    > Received: hello
"""



from __future__ import print_function
import socket
import argparse


BUFFER_SIZE = 1024
SERVER_PORT = 12345
MESSAGE = 'Hello TCP Server'


def tcp_client(message=MESSAGE):
    print('TCP Client')
    sock_client = socket.socket(socket.AF_INET,
                                socket.SOCK_STREAM)
    sock_client.connect(('localhost', SERVER_PORT))
    sock_client.send(message.encode())
    data = sock_client.recv(BUFFER_SIZE)
    sock_client.close()

    print('Sent    : ', message)
    print('Received: ', data)


def tcp_server():
    print('TCP Server')
    sock_listen = socket.socket(socket.AF_INET,
                                socket.SOCK_STREAM)
    sock_listen.bind(('localhost', SERVER_PORT))
    sock_listen.listen(1)
    conn, addr = sock_listen.accept()
    print(addr, ' connected')

    while True:
        data = conn.recv(BUFFER_SIZE)
        if not data:
            print('connection lost, stopping')
            break
        print('received data: ', data)
        conn.send(data)
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TCP socket example')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-s', '--server', action='store_true',
                       help='run example in server mode')
    group.add_argument('-c', '--client', type=str,
                       help='run example in client mode')
    args = parser.parse_args()

    print('args.server = ', args.server)
    print('args.client = ', args.client)

    if args.server:
        tcp_server()
    elif args.client:
        tcp_client(args.client)

