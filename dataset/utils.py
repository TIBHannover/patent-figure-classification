import json
import yaml

def get_config():
    """Get a config file"""
    return yaml.safe_load(open("config.yaml"))

def id_wrapper(idx):
    """Fill the id with 0s"""
    return f"F{str(idx).zfill(10)}"

def load_json(json_path):
    """Load JSON file"""
    with open(json_path, 'r') as f: return json.load(f)

def save_json(data, json_path):
    """Save JSON file"""
    with open(json_path, 'w') as f: json.dump(data, f, indent=4)

def get_patent_id(filename):
    """Return patent id"""
    if filename.startswith("EP"):
        if len(filename.split("-")) > 4:
            return "-".join(filename.split("-")[:-2])
        else:
            return "EP"+filename[7:14]
    elif filename.startswith("US"):
        return "US"+filename.split("-")[0].split("US")[1]
    elif filename.startswith("WO"):
        return "WO"+filename.split("img")[0][-10:]