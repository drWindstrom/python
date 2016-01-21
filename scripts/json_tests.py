import json
from collections import OrderedDict

jfile = 'test.json'
with open(jfile) as f:
    read_vars = json.load(f, object_pairs_hook=OrderedDict)
