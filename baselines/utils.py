import json
import yaml

def get_config():
    return yaml.safe_load(open("config.yaml"))

def load_json(file):
    with open(file, 'r') as f: return json.load(f)

def save_json(file, data):
    with open(file, 'w') as f: json.dump(data, f, indent=4)
