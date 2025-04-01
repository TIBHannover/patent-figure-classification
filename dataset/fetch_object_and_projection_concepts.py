import os
import random

from tqdm import tqdm

from utils import load_json, save_json, get_config

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_value = 1337
set_seed(seed_value)

CONFIG = get_config()
ROOT_DIR = CONFIG['root_dir']
WRITE_DIR = "concepts"

DEEPPATENT2_ROOT_DIR = f"{ROOT_DIR}/deeppatent2"
YEARS = range(2007, 2020+1, 1)

object_concepts = {"concepts": set()}
projection_concepts = {"concepts": set()}

for year in YEARS:

    raw_data = {}
    data = load_json(f"{DEEPPATENT2_ROOT_DIR}/{year}/design{year}.json")

    for sample in tqdm(data, desc=f"DeepPatent2/{year}"):

        if "aspect" in sample:
            if sample["aspect"]:
                projection_concepts["concepts"].add(sample["aspect"])

        # Object
        if "object" in sample:
            if sample["object"]:
                object_concepts["concepts"].add(sample["object"])

object_concepts["concepts"] = list(object_concepts["concepts"])
projection_concepts["concepts"] = list(projection_concepts["concepts"])

print(f"Total object concepts: {len(object_concepts['concepts'])}")
print(f"Total projection concepts: {len(projection_concepts['concepts'])}")

save_json(object_concepts, f"{WRITE_DIR}/object_concepts.json")
save_json(projection_concepts, f"{WRITE_DIR}/projection_concepts.json")