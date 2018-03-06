# saved as greeting-client.py
import Pyro4


def print_name_server_object_list():
    """
    Use name_server function
    """
    ns = Pyro4.locateNS(host='localhost')
    print(ns.lookup('maker'))
    print(ns.list())


maker = Pyro4.Proxy("PYRONAME:maker@localhost")
maker.greet('hello')
