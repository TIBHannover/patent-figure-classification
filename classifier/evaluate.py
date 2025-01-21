import os

import random
import logging

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import torch

from model_utils import get_model_and_preprocess

from dataset import get_dataloader

from metadata import get_options, get_template
from common_utils import save_json, get_config, EXPERIMENTS

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Aspect/Approach to Num of sample dictionary
N = {
    'type': {'bc': '54', 'mc-ts': '81', 'oc': '81'}, 
    'projection': {'bc': '81', 'mc-ts': '18', 'oc': '81'}, 
    'uspc': {'bc': '27', 'mc-ts': '81', 'oc': '54'},
    'object': {'bc': '150', 'mc-ts': '150', 'oc': '81'}
}

def main(classifier):

    config = get_config()

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    MODELS = config["models"]
    dataset = config["dataset"]

    for model_name in MODELS:

        for experiment_id, experiment in EXPERIMENTS[classifier].items():

            aspect = experiment.split("/")[0]
            template = get_template(
                            classifier, 
                            "object" if aspect == "object_held_out" else aspect
                        )

            if model_name == "flant5xl_zs_all_tasks":
                _model_name = "flant5xl_zs_all_tasks"
                model_dir = None
            else:
                _aspect = "object" if aspect == "object_held_out" else aspect
                _model_name = model_name.format(_aspect, N[_aspect][classifier])
                model_dir = f"LAVIS/lavis/experiment_outputs/instructblip/"
                model_dir += f"{dataset}_{_model_name}"

            print(f"Running experiment: {_model_name}/{classifier}/{experiment}")
                
            model, vis_preprocess = get_model_and_preprocess(model_dir)

            loader = get_dataloader(aspect, vis_preprocess)
            options = get_options(aspect, loader)

            if classifier == "bc":
                from binary_classifier import classify, compute_accuracy
                
                answers = classify(model, loader, template, options)
                compute_accuracy(answers)

            if classifier == "mc-ts":
                from mc_tournament_classifier import classify, compute_accuracy
                
                level = 0
                t = experiment.split("/")[1]
                t = int(t.split("O")[1])

                answers = classify(model, loader, template, options, t, level)
                compute_accuracy(answers)

            if classifier == "oc":
                from open_classifier import classify, compute_accuracy
                answers = classify(model, loader, template, aspect)
                compute_accuracy(answers)

            save_dir = f"results_latest/classification/{classifier}/{_model_name}/{experiment}"
            os.makedirs(save_dir, exist_ok=True)
            save_json(f"{save_dir}/answers.json", answers)

if __name__ == "__main__":

    for classifier in ["bc", "oc", "mc-ts"]:
        main(classifier)
