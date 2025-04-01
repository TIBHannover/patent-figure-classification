import os
import json
import random

from utils import load_json, save_json

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(1337)

RAW_DATASET_DIR = "raw_dataset"
YEARS = range(2007, 2020+1, 1)

INCLUDE_MISCALLENEOUS_CLASS = False

# DeepPatent2 Data
raw_data = {}
for year in YEARS:
    raw_data.update(load_json(f"{RAW_DATASET_DIR}/{year}.json"))

# CLEF IP Data
for split in ["train", "val", "test"]:
    raw_data.update(load_json(f"{RAW_DATASET_DIR}/clefip_{split}.json"))

# Aspect dictionaries
aspect2idx = {
    "type": load_json(f"{RAW_DATASET_DIR}/type2idx.json"),
    "projection": load_json(f"{RAW_DATASET_DIR}/projection2idx.json"),
    "object": load_json(f"{RAW_DATASET_DIR}/object2idx.json"),
    "uspc": load_json(f"{RAW_DATASET_DIR}/uspc2idx.json")
}

if not INCLUDE_MISCALLENEOUS_CLASS:
    del aspect2idx["uspc"]["miscellaneous"]

print(f"Total concepts")
for aspect in aspect2idx.keys():
    print(aspect, len(aspect2idx[aspect].keys()))

filteredObject2idx = {}
heldObject2idx = {}
for concept, samples in aspect2idx["object"].items():
    if len(samples) >= 150: filteredObject2idx[concept] = samples
    elif len(samples) >= 10: heldObject2idx[concept] = samples

aspect2idx["object"] = filteredObject2idx
aspect2idx["object_held_out"] = heldObject2idx

print(f"Total held-out concepts {len(heldObject2idx)}")

save_json(
        {"concepts": list(heldObject2idx.keys())},
        f"classification/object_held_out/all_held_concepts.json"
    )

def get_image_path(sample):
    """Prepare image path"""
    if sample['dir'] in ["train", "test", "val"]: # This is True for the CLEF-IP dataset
        return os.path.join("clef_ip", "classification", sample['dir'], sample['type'], sample['figure_file'])
    else:
        return os.path.join("deeppatent2", sample['dir'], sample['figure_file'])

def fetch_sample_data(id, aspect):
    """Prepare data sample"""
    sample = raw_data[id]

    return {
        "id": id,
        "patent_id": sample["patent_id"],
        "figure_file": sample["figure_file"],
        "figure_path": get_image_path(sample),
        aspect: sample["object" if aspect.startswith("object") else aspect]
    }

def create_test_set(aspect, split=None):
    """Create Test or Valid dataset"""
    test_dataset = {}

    concept2idx = aspect2idx[aspect][split] if aspect == "type" else aspect2idx[aspect]

    def sample_one_for_concept(concept, sample_ids, held_out=None):
        if len(sample_ids) < 1: return
        id = random.sample(sample_ids, 1)[0]
        test_dataset[id] = fetch_sample_data(id, aspect)
        if held_out: test_dataset[id]["held_out"] = True
        aspect2idx[aspect][concept].remove(id) # Remove test ids

    if aspect == "type":
        for concept, sample_ids in concept2idx.items():
            for id in sample_ids:
                test_dataset[id] = fetch_sample_data(id, aspect)
    else:
        if aspect == "object_held_out":
            # For object fetch 1 sample per random held concept
            concepts = random.sample(list(heldObject2idx.keys()), 500)
            for concept in concepts:
                sample_one_for_concept(concept, heldObject2idx[concept], held_out=True)
            save_json(
                {"concepts": concepts},
                f"classification/object_held_out/concepts.json"
            )
        else:
            # Fetch 1 sample per concept
            for concept, sample_ids in concept2idx.items():
                sample_one_for_concept(concept, sample_ids)

            # Sample randomly for remaining until 1000 (not true for object)
            while len(test_dataset) < 1000:
                concept = random.sample(list(concept2idx.keys()), 1)[0]
                if len(concept2idx[concept]) <= 150: continue # Preserve 150 samples for training
                sample_one_for_concept(concept, concept2idx[concept])

    return test_dataset

def create_train_set(aspect, split=None):
    """Create Train dataset"""
    train_dataset = {}

    concept2idx = aspect2idx[aspect]["train"] if aspect == "type" else aspect2idx[aspect]

    for _, sample_ids in concept2idx.items():

        if split == "train":
            _sample_ids = sample_ids
        else:
            sample_size = min(int(split.split("_")[-1]), len(sample_ids))
            _sample_ids = random.sample(sample_ids, sample_size) 

        for id in _sample_ids:
            train_dataset[id] = fetch_sample_data(id, aspect)
    
    return train_dataset

def create_datasets(aspect):
    """Create classification dataset for aspect"""

    for split in ["valid", "test", "train"]: # Run test and valid first
        
        dataset = create_train_set(aspect, split) if split.startswith("train") else create_test_set(aspect, split)
        
        print(f"{aspect}/{split}: {len(dataset)}")

        _aspect = aspect
        if aspect == "uspc" and INCLUDE_MISCALLENEOUS_CLASS:
            _aspect = "uspc_w_misc"

        os.makedirs(f"classification/{_aspect}", exist_ok=True)    
        with open(f"classification/{_aspect}/{split}.json", "w") as wf:
            json.dump(dataset, wf, indent=4)

if __name__ == "__main__":

    for aspect in ["type","projection","object","uspc"]:
        create_datasets(aspect)
