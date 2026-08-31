#!/usr/bin/env python3
import json
from cod_ssl.utils.runtime import runtime_info
print(json.dumps(runtime_info(), indent=2))

