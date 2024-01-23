import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset
from sklearn.metrics import f1_score

class TimeSeriesClassifier(LightningModule):
    def __init__(self, model, optimizer):
        super(TimeSeriesClassifier, self).__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss() if model.num_classes > 2 else nn.BCEWithLogitsLoss()
        self.opt = optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, labels.squeeze())
        y_hat = torch.argmax(logits, dim=1)
        acc = (y_hat == labels).float().mean()
        f1 = f1_score(labels.squeeze().cpu().numpy(), y_hat.cpu().numpy(), average='macro')
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_accuracy', acc, on_epoch=True)
        self.log("val_f1", f1, prog_bar=False, on_epoch=True) # type: ignore
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, labels.squeeze())
        y_hat = torch.argmax(logits, dim=1)
        acc = (y_hat == labels).float().mean()
        f1 = f1_score(labels.squeeze().cpu().numpy(), y_hat.cpu().numpy(), average='macro')
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_accuracy', acc, on_epoch=True)
        self.log("val_f1", f1, prog_bar=False, on_epoch=True) # type: ignore
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        y_hat = torch.argmax(logits, dim=1)
        acc = (y_hat == labels).float().mean()
        f1 = f1_score(labels.squeeze().cpu().numpy(), y_hat.cpu().numpy(), average='macro')
        self.log('accuracy', acc, on_epoch=True)
        self.log("f1", f1, prog_bar=False, on_epoch=True) # type: ignore
        return

    def configure_optimizers(self):
        return self.opt

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, label_mapping=None):
        self.X = X
        self.y = y
        self.label_mapping = label_mapping or self.create_label_mapping(y)
        self.y_mapped = [self.label_mapping[label] for label in y]


    def create_label_mapping(self, y):
        unique_labels = set(y)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        return label_mapping

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.Tensor(self.X[index]), torch.LongTensor([self.y_mapped[index]]).squeeze()