import random

from templates import templates
from common_utils import load_json, get_config

config = get_config()

num_workers = config["num_workers"]
batch_size = config["batch_size"]

_ASPECTS = ["type", "uspc", "object", "projection"]

concepts = {
    aspect: list(load_json(f"{config['cls_dataset_path']}/{aspect}/concepts.json")["concepts"])
    for aspect in _ASPECTS
}

def get_template(classifier, aspect):
    """
        Return 
    """
    if classifier == "bc":
        SELECTED_TEMPLATES_IDS = {
            "type": "T110",
            "projection": "T120",
            "object": "T130",
            "uspc": "T140"
        }
    elif classifier == "oc":
        SELECTED_TEMPLATES_IDS = {
            "type": "T310",
            "projection": "T320",
            "object": "T330",
            "uspc": "T340"
        }
    else:
        SELECTED_TEMPLATES_IDS = {
            "type": "T210",
            "projection": "T220",
            "object": "T230",
            "uspc": "T240"
        }
    
    return templates[SELECTED_TEMPLATES_IDS[aspect]]

def get_options(aspect, loader):

    options_dict = {}

    for batch in loader:

        for id in batch[0]:
            
            options = concepts[aspect]
            random.shuffle(options)

            options_dict[id] = {
                "concepts": options
            }

    return options_dict

