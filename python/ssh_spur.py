#!/usr/bin/env python3

"""
sudo apt install python3-spur
"""

import spur


host = "127.0.0.1"
username = "user"
password = "password"

shell = spur.SshShell(hostname=host, username=username, password=password,
                      missing_host_key=spur.ssh.MissingHostKey.accept,
                      connect_timeout=5)
response = shell.run(["ls"], allow_error=True)
