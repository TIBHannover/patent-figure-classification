
import json
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms as T
from torchvision.models import resnet50, ResNet50_Weights

import pytorch_lightning as L
import webdataset as wds

from dataset import EvalWebDataset

L.seed_everything(1337, workers=True)

model_transforms = ResNet50_Weights.IMAGENET1K_V2.transforms()

transforms_dict = {
    "train": T.Compose([
        T.RandomChoice([
            T.RandomHorizontalFlip(p=1.0),
            T.RandomVerticalFlip(p=1.0),
            T.RandomRotation((-45, 45), interpolation=T.InterpolationMode.BICUBIC, expand=True, fill=255),
            T.RandomRotation(degrees=90, interpolation=T.InterpolationMode.BICUBIC, expand=True, fill=255),
            T.RandomPerspective(distortion_scale=0.5, p=1.0, interpolation=T.InterpolationMode.BICUBIC, fill=255),
        ]),
        model_transforms
    ]),
    "val": model_transforms,
    "test": model_transforms
}

def setup_dataset(aspect):
    print(f"Setting up dataset for {aspect}")

    dataset_splits_dict = {
        split: EvalWebDataset(aspect=aspect,
                                split=split,
                                transform=transforms_dict[split])
                for split in ["train", "val", "test"]
        }
    
    return dataset_splits_dict, dataset_splits_dict["train"].get_concepts()

class ResNet(nn.Module):
    def __init__(self, num_target_classes):
        super(ResNet, self).__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()
        self.classifier = nn.Linear(backbone.fc.in_features, num_target_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x).flatten(1)
        x = self.classifier(features)
        return x

class ResNetModule(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.dataset_splits_dict, self.concepts = setup_dataset(self.aspect)
        self.num_classes = len(self.concepts)

        print(f"Num classes: {self.num_classes}")

        self.model = ResNet(self.num_classes)
        self.loss_module = nn.CrossEntropyLoss()
        
        self.save_hyperparameters()
        self.results = {}

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        imgs, concepts = batch[1], batch[2]
        preds = self.model(imgs)
        loss = self.loss_module(preds, concepts)
        
        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, concepts = batch[1], batch[2]
        preds = self.model(imgs)

        loss = self.loss_module(preds, concepts)
            
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        ids = batch[0]
        imgs, concepts = batch[1], batch[2]
        preds = self.model(imgs)
        acc = (preds.argmax(dim=-1) == concepts).float().mean()

        self.log('test_acc', acc)
    
    def train_dataloader(self, **kwargs):
        return wds.WebLoader(self.dataset_splits_dict["train"].get_wds(),
                            shuffle=False,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                            pin_memory=True,)
    
    def val_dataloader(self, **kwargs):
        return wds.WebLoader(self.dataset_splits_dict["val"].get_wds(),
                            shuffle=False,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                            pin_memory=True)
    
    def test_dataloader(self, **kwargs):
        return wds.WebLoader(self.dataset_splits_dict["test"].get_wds(),
                            shuffle=False,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                            pin_memory=True)
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), 
                              lr=self.lr,
                              momentum=self.momentum)

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.first_restart
        )

        return [optimizer], [scheduler]

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        model_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        self.model.load_state_dict(model_state_dict)
    
    def save_results(self):
        with open(f"results/classification/{self.aspect}/rn50/answers.json", "w") as wf:
            json.dump(self.results, wf)
