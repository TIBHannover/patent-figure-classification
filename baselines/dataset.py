import os
import io

from PIL import Image

import torchvision.transforms as T

from braceexpand import braceexpand
import webdataset as wds

from utils import load_json, get_config

config = get_config()

class EvalWebDataset:

    def __init__(
            self,
            aspect,
            split,
            transform=None
    ):
        self.root = config['root_dir']
        self.aspect = aspect
        self.transform = transform if transform else T.Compose([lambda x: x])
        self.split = split + "_150" if split == "train" else split
        data_dir = f"{self.root}/deeppatent2/cls/{aspect}/{self.split}/"
        shard_count = f"{(len(os.listdir(data_dir))-1):06d}"
        self.shards = braceexpand(data_dir+"shard-{000000.."+shard_count+"}.tar")

        self.concept2idx = {concept: idx for idx, concept in enumerate(self.get_concepts())}
        
    def get_concepts(self):
        return load_json(f"{config['cls_dataset_path']}/{self.aspect}/concepts.json")["concepts"]

    def get_wds(self):
        return (wds.WebDataset(self.shards)
                    .shuffle(5000 if self.split.startswith("train") else 0)
                    .to_tuple("__key__", "image.png", "label.txt")
                    .map_tuple(
                        lambda key: key,
                        lambda image: self.transform(Image.open(io.BytesIO(image))),
                        lambda label: self.concept2idx[label.decode("utf-8")]
                    ))
