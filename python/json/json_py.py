#!/usr/bin/env python3

"""
Example json project

Author: Zihan Chen
Date: 2022-12-21
"""

import json

# local string
y = '{ "name":"John", "age":30, "city":"New York"}'
cfg = json.loads(y)
print('Name: {}'.format(cfg['name']))

# convert a Python dict to json
y_dict = {
  "name": "John",
  "age": 30,
  "city": "New York"
}
y_json = json.dumps(y_dict, indent=4)

# save to a file
with open("example.json", "w") as out:
    out.write(y_json)

# read from a file
f = open('example.json', 'r')
y_json_read = json.load(f)
print("Name read: {}".format(y_json_read['name']))
