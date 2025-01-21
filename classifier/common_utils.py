import json
import yaml

def get_config():
    return yaml.safe_load(open("classifier/config.yaml"))

config = get_config()

ASPECTS = config["aspects"]
T = config["options_count"]

def load_json(file):
    """
        Load from a JSON file

        Args:
            file: JSON file path
        
        Returns:
            JSON dictionary
    """

    with open(file, 'r') as f: return json.load(f)

def save_json(file, data):
    """
        Save data as JSON file

        Args:
            file: JSON file path
            data: data in dictionary format to save
            
        Returns:
            None
    """
    with open(file, 'w') as f: json.dump(data, f, indent=4)

EXPERIMENTS = {
    "traditional": {},
    "bc": {},
    "oc": {},
    "mc-ts": {}
}

experiment_id = 1

for aspect in ASPECTS:
    experiment = f"{aspect}/"
    EXPERIMENTS["traditional"][experiment_id] = experiment
    EXPERIMENTS["bc"][experiment_id] = experiment
    EXPERIMENTS["oc"][experiment_id] = experiment
    experiment_id += 1

for aspect in ASPECTS:
    for t in T:
        _t = t
        if aspect in ["type", "projection"] and _t == 20: continue

        experiment = f"{aspect}/O{_t}/"
        EXPERIMENTS["mc-ts"][experiment_id] = experiment
        experiment_id += 1
