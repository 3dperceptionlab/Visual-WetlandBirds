import torch
import pytorch_lightning as pl
from torchvision.models.video import r3d_18, mvit_v2_s, s3d, swin3d_s
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import wandb
import sys
sys.path.append('modules')

from model import VideoClassificationModel
from dataset import get_dataloader, id_to_label
import torch.nn as nn

torch.set_float32_matmul_precision('medium')
pl.seed_everything(42, workers=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_class_weights(dataloader, num_classes):
    class_counts = torch.zeros(num_classes).to(device)
    for _, label in dataloader:
        label = label.view(-1).to(device)
        class_counts[label] += 1
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    return class_weights

def train():
    wandb.init(entity='CHAN-TWiN', project='JaviPuto', config={"_disable_artifacts": True})
    config = wandb.config

    train_loader, val_dataloader, test_dataloader = get_dataloader(batch_size=config.batch_size, shuffle=True)
    class_weights = torch.tensor([0.0849, 0.5297, 0.1976, 0.3531, 0.2488, 5.1205, 0.4655]).to(device)
    print(f"Class weights: {class_weights}")

    if config.pretrained_model:
        if config.model == 'r3d_18':
            model = r3d_18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, len(id_to_label))
        elif config.model == 'mvit_v2_s':
            model = mvit_v2_s(pretrained=True)
            model.head[1] = nn.Linear(model.head[1].in_features, len(id_to_label))
        elif config.model == 's3d':
            model = s3d(pretrained=True)

            model.classifier.add_module('2', nn.Flatten())
            model.classifier.add_module('3', nn.Linear(400, len(id_to_label)))

            # Modify the forward method
            def new_forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = self.classifier(x)  # Directly use the classifier
                return x  # Output already has shape (batch_size, num_classes)

            model.forward = new_forward.__get__(model, type(model))
            
        elif config.model == 'swin3d_s':
            model = swin3d_s(pretrained=True)
            model.head = nn.Linear(model.head.in_features, len(id_to_label))
        else:
            raise ValueError(f"Model {config.model} not supported")
    else:
        if config.model == 'r3d_18':
            model = r3d_18(weights=None, num_classes=len(id_to_label))
        elif config.model == 'mvit_v2_s':
            model = mvit_v2_s(weights=None, num_classes=len(id_to_label))
        elif config.model == 's3d':
            model = s3d(weights=None, num_classes=len(id_to_label))
        elif config.model == 'swin3d_s':
            model = swin3d_s(weights=None, num_classes=len(id_to_label))
        else:
            raise ValueError(f"Model {config.model} not supported")


    model = model.to(device)

    lightning_model = VideoClassificationModel(model=model, learning_rate=config.learning_rate, loss_weight=class_weights if config.class_weight else None)

    wandb_logger = WandbLogger(entity='CHAN-TWiN', project='JaviPuto', log_model=True)

    early_stopping = EarlyStopping(monitor="val_accuracy_epoch", patience=15, mode="max")

    trainer = pl.Trainer(max_epochs=200, logger=wandb_logger, log_every_n_steps=5, callbacks=[early_stopping])
    trainer.fit(lightning_model, train_loader, val_dataloader)
    trainer.test(lightning_model, test_dataloader)

sweep_config = {
    'method': 'random',
    'metric': {'name': 'test_accuracy_epoch', 'goal': 'maximize'},
    'pretrained_model': {'values': [True, False]},
    'parameters': {
        'learning_rate': {'distribution': 'uniform', 'min': 0.0001, 'max': 0.01},
        'batch_size': {'values': [1, 2, 4]},
        'model': {'values': ['s3d']},
        'class_weight': {'values': [True, False]},
    },
    'secondary_metrics': ['val_f1_epoch', 'val_precision_epoch', 'val_recall_epoch', 'test_f1_epoch', 'test_precision_epoch', 'test_recall_epoch', 'val_accuracy_epoch', 'test_accuracy_epoch', 'train_accuracy_epoch']
}

sweep_id = wandb.sweep(sweep_config, entity='CHAN-TWiN', project='JaviPuto')
wandb.agent(sweep_id, entity='CHAN-TWiN', project='JaviPuto', function=train)