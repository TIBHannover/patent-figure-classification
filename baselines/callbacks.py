from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


callbacks = [
    LearningRateMonitor("epoch"),
    ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss", save_last=True, save_top_k=3),         
]
