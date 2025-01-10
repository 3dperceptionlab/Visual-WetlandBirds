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

from sklearn.metrics import confusion_matrix

checkpoint = '/dataset/checkpoints/baseline.ckpt'

def calculate_class_weights(dataloader, num_classes):
    class_counts = torch.zeros(num_classes).to(device)
    for _, label in dataloader:
        label = label.view(-1).to(device)
        class_counts[label] += 1
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    return class_weights

def train():

    pretrained_model = False
    learning_rate = 0.0026901178094969895
    batch_size = 4
    model_name = 'r3d_18'
    class_weight = False

    _, _, test_dataloader = get_dataloader(batch_size=batch_size, shuffle=True)
    class_weights = torch.tensor([0.0849, 0.5297, 0.1976, 0.3531, 0.2488, 5.1205, 0.4655]).to(device)
    print(f"Class weights: {class_weights}")

    if pretrained_model:
        if model_name == 'r3d_18':
            model = r3d_18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, len(id_to_label))
        elif model_name == 'mvit_v2_s':
            model = mvit_v2_s(pretrained=True)
            model.head[1] = nn.Linear(model.head[1].in_features, len(id_to_label))
        elif model_name == 's3d':
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
            
        elif model_name == 'swin3d_s':
            model = swin3d_s(pretrained=True)
            model.head = nn.Linear(model.head.in_features, len(id_to_label))
        else:
            raise ValueError(f"Model {model_name} not supported")
    else:
        if model_name == 'r3d_18':
            model = r3d_18(weights=None, num_classes=len(id_to_label))
        elif model_name == 'mvit_v2_s':
            model = mvit_v2_s(weights=None, num_classes=len(id_to_label))
        elif model_name == 's3d':
            model = s3d(weights=None, num_classes=len(id_to_label))
        elif model_name == 'swin3d_s':
            model = swin3d_s(weights=None, num_classes=len(id_to_label))
        else:
            raise ValueError(f"Model {model_name} not supported")


    model = model.to(device)

    lightning_model = VideoClassificationModel.load_from_checkpoint(checkpoint_path=checkpoint, model=model, learning_rate=learning_rate, loss_weight=class_weights if class_weight else None)
    lightning_model = lightning_model.to(device)

    wandb_logger = WandbLogger(entity='CHAN-TWiN', project='JaviPuto', log_model=True)

    trainer = pl.Trainer(max_epochs=200, logger=wandb_logger, log_every_n_steps=5)

    lightning_model.eval()  # Set the model to evaluation mode
    lightning_model.model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation for testing
        for batch in test_dataloader:
            inputs, labels = batch

            outputs = lightning_model(inputs)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Generate the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    trainer.test(lightning_model, test_dataloader)

train()