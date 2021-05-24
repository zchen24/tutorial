#!/usr/bin/env python3


import subprocess

cmd = ["python3", "return_code.py"]
p = subprocess.Popen(cmd)
ret_code = p.wait()

if ret_code != 0:
    exit(ret_code)
else:
    exit(0)
