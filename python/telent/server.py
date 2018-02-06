#!/usr/bin/env python

from miniboa import TelnetServer
import logging


"""
Telnet server using miniboa
Zihan Chen 2018-02-06 
"""

clients = []

def on_connect(client):
    clients.append(client)
    logging.debug('client ' + client.address + ':' +
                  str(client.port) + ' connected')

    
def on_disconnect(client):
    clients.remove(client)
    logging.debug('client disconnected')


def process_client():
    for client in clients:
        if client.active and client.cmd_ready:
            cmd = client.get_command()
            logging.debug('command = ' + cmd)

            if cmd.lower() == 'alibaba':
                client.send('Door openning\n')
            else:
                client.send('OK\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    server = TelnetServer(
        port=9876,
        on_connect=on_connect,
        on_disconnect=on_disconnect)
    logging.info("Started Telnet server on port {}.".format(server.port))
    
    while True:
        server.poll()
        process_client()
