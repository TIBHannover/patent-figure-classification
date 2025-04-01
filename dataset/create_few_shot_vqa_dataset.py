import os
import random
from templates import templates

from utils import load_json, save_json

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(1337)

ASPECTS = ["type", "projection", "uspc", "object"]
ASPECTS = ["object_held_out"]
TASKS = ["bc", "mc", "oc"]
MC_OPTIONS = [5, 10, 20]
SPLITS = ["train", "val", "test"]
N = [3, 6, 9, 18, 27, 50]

def get_template(task, aspect):
    """Get the corresponding template for the given task and aspect"""
    if task == "bc":
        SELECTED_TEMPLATES_IDS = {
            "type": "T110",
            "projection": "T120",
            "object": "T130",
            "uspc": "T140"
        }
    elif task == "oc":
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

concepts = {
    "type": load_json(f"classification/type/concepts.json")["concepts"],
    "projection": load_json(f"classification/projection/concepts.json")["concepts"],
    "uspc": load_json(f"classification/uspc/concepts.json")["concepts"],
    "object": load_json(f"classification/object/concepts.json")["concepts"],
    "object_held_out": load_json(f"classification/object_held_out/concepts.json")["concepts"],
}

top_n_similar_concepts = {
    "type": concepts["type"],
    "projection": concepts["projection"],
    "uspc": concepts["uspc"],
    "object": load_json(f"concepts/top_n_similar_objects.json"),
    "object_held_out": load_json(f"concepts/top_n_similar_object_held_out.json"),
}

def create_binary_question(template, concept):
    """Create binary question from a template"""
    question = template["question"]
    aspect = template["aspect"]

    answer = random.sample(["yes", "no"], 1)[0]
    if answer == "no":
        options = set(top_n_similar_concepts[aspect]) - set([concept])
        concept = random.sample(list(options), 1)[0] # Select a random concept

    question = question.replace("{"+f"{aspect}"+"}", concept)

    return question, answer

def create_multiple_choice_question(template, concept, n_option):
    """Create a multiple choice question from a template."""
    
    question = template["question"]
    aspect = template["aspect"]
    
    options = set(top_n_similar_concepts[aspect]) - set([concept])                # Remove correct concept
    options_list = random.sample(list(options), min(n_option-1, len(options)))  # Sample a random concept
    options_list.append(concept)                                                # Add correct concept
    random.shuffle(options_list)
    
    options_text = [f"({i+1}) {options_list[i]}" for i in range(len(options_list))]
    question = question.replace("{options_text}", " ".join(options_text))
    
    return question, options_list.index(concept)+1

def create_open_ended_question(template):
    """Create open ended question from a template"""
    question = template["question"]
    return question

def prepare_vqa_sample(aspect, task, sample):
    """Prepare a VQA data sample"""

    if aspect == "object_held_out": aspect = "object"

    template = get_template(task, aspect)

    if task == "bc":
        question, answer = create_binary_question(template, sample[aspect])
        sample.update({
            "task": task,
            "question": question,
            "answer": str(answer)
        })
        return sample
    elif task == "oc":
        question = create_open_ended_question(template)
        sample.update({
            "task": task,
            "question": question,
            "answer": str(sample[aspect])
        })
        return sample
    else:
        n_option = int(task.split("-")[-1])
        question, answer = create_multiple_choice_question(template, sample[aspect], n_option)
        sample.update({
            "task": task,
            "question": question,
            "answer": str(answer)
        })
        return sample

for aspect in ASPECTS:

    for split in ["test"]:
        
        cls_dataset = load_json(f"classification/{aspect}/{split}.json")
        
        for task in TASKS:
            
            _tasks = [task]
            
            if task == "mc":
                _tasks = [f"{task}-{mc_option}" for mc_option in MC_OPTIONS]
                if aspect in ["type", "projection"]:
                    _tasks.remove("mc-20")
            
            for _task in _tasks:
                
                vqa_dataset = {}
                
                for id, sample in cls_dataset.items():
                    vqa_dataset[id] = prepare_vqa_sample(aspect, _task, sample)
                
                os.makedirs(f"vqa/{aspect}/{_task}/", exist_ok=True)
                save_json(vqa_dataset, f"vqa/{aspect}/{_task}/{split}.json")

    # Create Train dataset
    cls_dataset = load_json(f"classification/{aspect}/train_150.json")

    concept2idx = {}
    for id, sample in cls_dataset.items():
        concept2idx.setdefault(sample[aspect], [])
        concept2idx[sample[aspect]].append(id)
    
    for n in N:
        
        concept2idx_per_task = {}

        for concept in concept2idx.keys():
            
            sample_ids = concept2idx[concept][:min(n*3, 150)] # n = 3, n*3=9
            random.shuffle(sample_ids)

            for i, task in enumerate(TASKS):
                concept2idx_per_task.setdefault(task, {})
                concept2idx_per_task[task].setdefault(concept, [])
                concept2idx_per_task[task][concept] = sample_ids[i*n:i*n+n]

        for task in concept2idx_per_task.keys():
            
            if task == "mc":
                
                mc_options = MC_OPTIONS[:-1] if aspect in ["type", "projection"] else MC_OPTIONS
                
                if len(mc_options) == 3: # Object, USPC
                    split_at = int(n/len(mc_options))
                    if n == 50: split_at += 1 # 17, 17, 16
                else:
                    split_at = int(n/len(mc_options))
                    if not n % 2 == 0: split_at += 1
                    
                for i, mc_option in enumerate(mc_options):
                    task = f"mc-{mc_option}"
                    vqa_dataset = {}   

                    for concept, ids in concept2idx_per_task["mc"].items():
                        id_splits = [ids[split_at*t:min(split_at*(t+1), len(ids))] for t in range(3)]
                        
                        for id in id_splits[i]:
                            
                            if len(vqa_dataset) == 1: print(f"{aspect}/{task}/{n}/{len(id_splits[i])}")

                            sample = cls_dataset[id]
                            vqa_dataset[id] = prepare_vqa_sample(aspect, task, sample)

                    os.makedirs(f"vqa/{aspect}/{task}/", exist_ok=True)
                    save_json(vqa_dataset, f"vqa/{aspect}/{task}/train_{min(n*3, 150)}.json")
            else:
                vqa_dataset = {}
                
                for concept, ids in concept2idx_per_task[task].items():
                    
                    for id in ids:

                        if len(vqa_dataset) == 1: print(f"{aspect}/{task}/{n}/{len(ids)}")
                        
                        sample = cls_dataset[id]
                        vqa_dataset[id] = prepare_vqa_sample(aspect, task, sample)
                
                os.makedirs(f"vqa/{aspect}/{task}/", exist_ok=True)
                save_json(vqa_dataset, f"vqa/{aspect}/{task}/train_{min(n*3, 150)}.json")