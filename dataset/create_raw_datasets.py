import os
import re
import csv
import random

from tqdm import tqdm

from utils import id_wrapper, load_json, save_json, get_patent_id, get_config

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(1337)

CONFIG = get_config()
ROOT_DIR = CONFIG['root_dir']
WRITE_DIR = "raw_dataset"

os.makedirs(WRITE_DIR, exist_ok=True)

concept2projection = load_json("concepts/concept2projection.json")
concept2object = load_json("concepts/concept2object.json")
uspc2title = load_json("concepts/uspc2title.json")

def fetch_projection(concept):
    """Fetch projection label"""
    def preprocess_concept(concept):
        concept = concept.encode('utf-8').decode('unicode_escape')
        concept = re.sub(r'\d+', '', concept)
        concept = re.sub(r'[^\w\s]', '', concept.lower()).strip()
        concept = concept.replace(" view", "")
        return concept
    
    try:
        return concept2projection[preprocess_concept(concept)].lower()
    except (KeyError, AttributeError):
        return None
    
def fetch_object(concept):
    """Fetch object label"""
    def preprocess_concept(concept):
        concept = concept.encode('utf-8').decode('unicode_escape')
        concept = concept.lower()
        concept = re.sub(r'\d+', '', concept)
        concept = re.sub(r'\s+', ' ', concept)
        concept = re.sub(r'[^\w\s]', '', concept).strip()
        return concept
    
    try:
        return concept2object[preprocess_concept(concept)].lower()
    except (KeyError, AttributeError):
        return None

def format_uspc(uspc):
    """Format USPC class code"""
    uspc = uspc.replace("  ", " ")
    if " " in uspc: 
        if uspc[1] == " ": # D 8403 -> D08 403
            mainclass = uspc.split(" ")[0] + "0" + uspc.split(" ")[1][0]
            if len(uspc.split(" ")) > 2:
                subclass = uspc.split(" ")[2]
            else:
                subclass = uspc.split(" ")[1][1:]
        else:
            mainclass, subclass = uspc.split(" ") # D12 91
    else:
        mainclass, subclass = uspc[:3], uspc[3:] # D13133 -> D13 133
    
    if len(subclass) > 3: subclass = subclass[:3] + "." + subclass[3:]
    return mainclass, subclass

def fetch_uspc_title(uspc):
    """Fetch USPC title given a USPC class code"""
    try:
        if "D" in uspc[0]:
            return uspc2title[uspc[0]]["title"].lower()
    except (KeyError, AttributeError):
        return None
    return None

def prepare_clef_ip_dataset(idx):
    """Prepare CLEF-IP dataset"""

    CLEF_IP_ROOT_DIR = f"{ROOT_DIR}/clef_ip/classification"

    type2idx = {}

    for split in ["train", "val", "test"]:
        
        raw_data = {}
        type2idx.setdefault(split, {})
        
        with open(f"{CLEF_IP_ROOT_DIR}/clean_{split}.csv", "r") as f:

            for type, figure in tqdm(csv.reader(f), desc=f"CLEF-IP 2011/{split}"):
                raw_data[id_wrapper(idx)] = {
                    "patent_id": get_patent_id(figure),
                    "figure_file": figure,
                    "dir": split,
                    "type": type.lower()
                }
                type2idx[split].setdefault(type, []).append(id_wrapper(idx))
                idx += 1

        save_json(raw_data, f"{WRITE_DIR}/clefip_{split}.json")
    
    save_json(type2idx, f"{WRITE_DIR}/type2idx.json")

    return idx

def prepare_deeppatent2_dataset(idx):
    """Prepare DeepPatent2 dataset"""

    DEEPPATENT2_ROOT_DIR = f"{ROOT_DIR}/deeppatent2"
    YEARS = range(2007, 2020+1, 1)

    projection2idx = {}
    object2idx = {}
    uspc2idx = {}

    for year in YEARS:
    
        raw_data = {}
        data = load_json(f"{DEEPPATENT2_ROOT_DIR}/{year}/design{year}.json")

        for sample in tqdm(data, desc=f"DeepPatent2/{year}"):
            
            # Projection
            if "aspect" in sample:
                if sample["aspect"]:
                    projection = fetch_projection(sample["aspect"])
            else:
                projection = None

            # Object
            if "object" in sample:
                if sample["object"]:
                    object = fetch_object(sample["object"])
            else:
                object = None

            # USPC
            if "classification_national" in sample:
                if sample["classification_national"]:
                    uspc_id = format_uspc(sample["classification_national"])
                    uspc_title = fetch_uspc_title(uspc_id)
            else:
                uspc_id, uspc_title = None, None
            
            projection = projection if projection else "None"
            object = object if object else "None"
            uspc_id = uspc_id[0] + "/" + uspc_id[1] if uspc_id else "None"
            uspc_title = uspc_title if uspc_title else "None"

            raw_data[id_wrapper(idx)] = {
                "patent_id": sample["patentID"],
                "figure_file": sample["subfigure_file"],
                "dir": f"{year}/Segmentednew/",
                "year": year,
                "projection": projection,
                "object": object,
                "uspc_id": uspc_id,
                "uspc": uspc_title
            }

            projection2idx.setdefault(projection, []).append(id_wrapper(idx))
            object2idx.setdefault(object, []).append(id_wrapper(idx))
            uspc2idx.setdefault(uspc_title, []).append(id_wrapper(idx))

            idx += 1

        save_json(raw_data, f"{WRITE_DIR}/{year}.json")
    
    del projection2idx["None"]
    del object2idx["None"]
    del uspc2idx["None"]

    save_json(projection2idx, f"{WRITE_DIR}/projection2idx.json")
    save_json(object2idx, f"{WRITE_DIR}/object2idx.json")
    save_json(uspc2idx, f"{WRITE_DIR}/uspc2idx.json")

if __name__ == "__main__":

    idx = 1

    last_idx = prepare_clef_ip_dataset(idx)
    prepare_deeppatent2_dataset(last_idx)
