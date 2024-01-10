import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset

class TimeSeriesClassifier(LightningModule):
    def __init__(self, model, lr):
        super(TimeSeriesClassifier, self).__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        # print(inputs.shape)
        logits = self(inputs)
        loss = self.loss_fn(logits, labels.squeeze())
        y_hat = torch.argmax(logits, dim=1)
        acc = (y_hat == labels).float().mean()
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_accuracy', acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, labels.squeeze())
        y_hat = torch.argmax(logits, dim=1)
        acc = (y_hat == labels).float().mean()
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_accuracy', acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

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