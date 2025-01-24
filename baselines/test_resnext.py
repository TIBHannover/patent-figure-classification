import os
import time
import json

from argparse import ArgumentParser

import torch

import pytorch_lightning as L

from resnext_101 import ResNetModule
from callbacks import callbacks

torch.set_float32_matmul_precision('medium')
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

L.seed_everything(1337, workers=True)

best_model_path = {
    "type": "path/to/checkpoint",
    "uspc": "path/to/checkpoint",
    "projection": "path/to/checkpoint",
    "object": "path/to/checkpoint"
}

def test_model(**kwargs):

    for aspect in best_model_path.keys():
        kwargs["aspect"] = aspect

        model = ResNetModule.load_from_checkpoint(best_model_path[aspect])
        model.batch_size = 64
        model.num_workers = 16

        idx2concept = {idx: concept for idx, concept in enumerate(model.concepts)}

        accuracy = 0.0
        accuracy_100 = 0.0

        batch_count = 0

        top_100 = {}

        os.makedirs(f"results/classification/rx101/{aspect}", exist_ok=True)
        with open(f"results/classification/rx101/{aspect}/answers.json", "w") as wf:

            for batch in model.test_dataloader():

                ids, imgs, concepts = batch

                imgs = imgs.to("cuda")
                concepts = concepts.to("cuda")

                preds = model(imgs)
                
                acc = (preds.argmax(dim=-1) == concepts).float().mean()
                accuracy += acc

                top_k = min(100, model.num_classes)
                _, top_k_preds = preds.topk(top_k, dim=1)
                top_k_correct = (top_k_preds == concepts.view(-1, 1)).float()
                
                acc = top_k_correct.sum(dim=1).mean()  # Average over the batch
                accuracy_100 += acc
                batch_count += 1

                for id, reference_answer, selected_answer, top_100_concepts in zip(ids, concepts, preds.argmax(dim=-1), top_k_preds):
                    id = id.item() if not isinstance(id, str) else id
                    top_100[id] = {
                        "selected_answer": idx2concept[selected_answer.item()],
                        "reference_answer": idx2concept[reference_answer.item()],
                        "top_100": [idx2concept[l.item()] for l in top_100_concepts]
                    }

            json.dump(top_100, wf, indent=4)


        print("Accuracy: ", (accuracy.item() / batch_count) * 100)
        print("Accuracy 100: ", (accuracy_100.item() / batch_count) * 100)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--model_name", default="resnext101_imagenetv1")
    parser.add_argument("--checkpoint_path", default="checkpoints")

    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--max_epochs", default=30)
    parser.add_argument("--num_workers", default=16)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--first_restart", default=5)

    kwargs = vars(parser.parse_args())

    test_model(**kwargs)
