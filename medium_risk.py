
import os
import json

def read_config():
    """Read configuration file"""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

config = read_config()
print("Config loaded")