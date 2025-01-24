import os
import time
from argparse import ArgumentParser

import torch

import pytorch_lightning as L

from resnext_101 import ResNetModule
from callbacks import callbacks

torch.set_float32_matmul_precision('medium')
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

L.seed_everything(1337, workers=True)

def train_model(**kwargs):
    
    aspect = "object"
    kwargs["aspect"] = aspect

    model_name = kwargs.get("model_name")
    CHECKPOINT_PATH = kwargs.get("checkpoint_path")

    trainer = L.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, model_name, aspect),
                        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                        max_epochs=kwargs.get("max_epochs"),
                        callbacks=callbacks,
                        enable_progress_bar=True)

    model = ResNetModule(**kwargs)

    print("Training...")
    trainer.fit(model,
                train_dataloaders=model.train_dataloader(),
                val_dataloaders=model.val_dataloader())

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--model_name", default="resnext101_imagenetv1")
    parser.add_argument("--checkpoint_path", default="checkpoints")

    parser.add_argument("--batch_size", default=256)
    parser.add_argument("--max_epochs", default=30)
    parser.add_argument("--num_workers", default=16)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--first_restart", default=5)

    kwargs = vars(parser.parse_args())

    train_model(**kwargs)
