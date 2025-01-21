import os
import io

from PIL import Image

import torch
import torchvision.transforms as T

from braceexpand import braceexpand
import webdataset as wds

from common_utils import get_config

config = get_config()

num_workers = config["num_workers"]
batch_size = config["batch_size"]
torch.manual_seed(config["seed"])

class EvalWebDataset:
    """PatFigCLS dataset class"""
    def __init__(
            self,
            aspect,
            split,
            transform=None
    ):
        self.root = "/nfs/data/vip_at_scale"
        self.aspect = aspect
        self.transform = transform if transform else T.Compose([lambda x: x])
        self.split = split + "_150" if split == "train" else split
        data_dir = f"cls2/{aspect}/{self.split}/"
        shard_count = f"{(len(os.listdir(data_dir))-1):06d}"
        self.shards = braceexpand(data_dir+"shard-{000000.."+shard_count+"}.tar")

    def get_wds(self):
        """
            Return the dataset as webdataset
        """
        return (wds.WebDataset(self.shards)
                    .shuffle(0)
                    .to_tuple("__key__", "image.png", "label.txt")
                    .map_tuple(
                        lambda key: key,
                        lambda image: self.transform(Image.open(io.BytesIO(image))),
                        lambda label: label.decode("utf-8")
                    ))

def get_dataloader(aspect, transform):
    """
        Return WebLoader for PatFIGCLS dataset
    """
    eval_dataset = EvalWebDataset(aspect, split="test", transform=transform)
                    
    loader = wds.WebLoader(eval_dataset.get_wds(),
                shuffle=False, num_workers=num_workers,
                batch_size=batch_size, pin_memory=True)
    
    return loader
