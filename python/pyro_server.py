#!/usr/bin/env python

"""
Pyro4 server example

To Run:

# start pyro4 name server
$ pyro4-ns

# start server
$ python pyro_server.py

# start client
$ python pyro_client.py
"""


import Pyro4


@Pyro4.expose
class Maker(object):
    def greet(self, name):
        print('Greeting from %s' % name)


if __name__ == '__main__':
    Pyro4.Daemon.serveSimple(
        {
            Maker: 'maker'
        },
        host='localhost'
    )
