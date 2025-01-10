import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.models.video import r3d_18
from pytorch_lightning.loggers import WandbLogger
import wandb
from torchmetrics import Accuracy, F1Score, Precision, Recall
from dataset import id_to_label


class VideoClassificationModel(pl.LightningModule):
    def __init__(self, model, learning_rate, loss_weight=None):
        super(VideoClassificationModel, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_weight = loss_weight

        num_classes = len(id_to_label)
        
        # Initialize metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        # F1, Precision, and Recall metrics for validation and test
        self.val_f1 = F1Score(num_classes=num_classes, average='macro', task='multiclass')
        self.val_precision = Precision(num_classes=num_classes, average='macro', task='multiclass')
        self.val_recall = Recall(num_classes=num_classes, average='macro', task='multiclass')

        self.test_f1 = F1Score(num_classes=num_classes, average='macro', task='multiclass')
        self.test_precision = Precision(num_classes=num_classes, average='macro', task='multiclass')
        self.test_recall = Recall(num_classes=num_classes, average='macro', task='multiclass')

    def forward(self, x):
        return self.model(x)

    def _calculate_loss_and_accuracy(self, outputs, labels, split):
        loss = F.cross_entropy(outputs, labels, weight=self.loss_weight)
        preds = torch.argmax(outputs, dim=1)

        if split == "train":
            accuracy = self.train_accuracy(preds, labels)
        elif split == "val":
            accuracy = self.val_accuracy(preds, labels)
            self.val_f1.update(preds, labels)
            self.val_precision.update(preds, labels)
            self.val_recall.update(preds, labels)
        else:  # test
            accuracy = self.test_accuracy(preds, labels)
            self.test_f1.update(preds, labels)
            self.test_precision.update(preds, labels)
            self.test_recall.update(preds, labels)

        self.log(f"{split}_loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log(f"{split}_accuracy_step", accuracy, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self._calculate_loss_and_accuracy(outputs, labels, "train")
        return loss

    def on_train_epoch_end(self):
        train_accuracy_epoch = self.train_accuracy.compute()
        self.log("train_accuracy_epoch", train_accuracy_epoch, on_epoch=True,  prog_bar=True)
        
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        self._calculate_loss_and_accuracy(outputs, labels, "val")

    def on_validation_epoch_end(self):
        # Compute and log epoch-level validation metrics
        val_accuracy_epoch = self.val_accuracy.compute()
        self.log("val_accuracy_epoch", val_accuracy_epoch, prog_bar=True)
        
        val_f1_epoch = self.val_f1.compute()
        val_precision_epoch = self.val_precision.compute()
        val_recall_epoch = self.val_recall.compute()

        # Log additional metrics
        self.log("val_f1_epoch", val_f1_epoch, on_epoch=True, prog_bar=True)
        self.log("val_precision_epoch", val_precision_epoch, on_epoch=True,  prog_bar=True)
        self.log("val_recall_epoch", val_recall_epoch, on_epoch=True,  prog_bar=True)

        # Reset validation metrics for the next epoch
        self.val_accuracy.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        self._calculate_loss_and_accuracy(outputs, labels, "test")

    def on_test_epoch_end(self):
        test_accuracy_epoch = self.test_accuracy.compute()
        self.log("test_accuracy_epoch", test_accuracy_epoch, prog_bar=True)
        
        test_f1_epoch = self.test_f1.compute()
        test_precision_epoch = self.test_precision.compute()
        test_recall_epoch = self.test_recall.compute()

        # Log additional metrics
        self.log("test_f1_epoch", test_f1_epoch, on_epoch=True, prog_bar=True)
        self.log("test_precision_epoch", test_precision_epoch, on_epoch=True, prog_bar=True)
        self.log("test_recall_epoch", test_recall_epoch, on_epoch=True, prog_bar=True)

        # Reset test metrics for the next epoch
        self.test_accuracy.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)