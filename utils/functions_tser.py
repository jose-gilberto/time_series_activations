import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset

class TimeSeriesRegressor(LightningModule):
    def __init__(self, model, lr):
        super(TimeSeriesRegressor, self).__init__()
        self.model = model
        self.loss_fn_mse = nn.MSELoss()
        self.loss_fn_mae = nn.L1Loss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        y_hat = self(inputs)
        # print('train', inputs.shape, labels.shape, y_hat.shape)

        # Mean Squared Error (MSE) loss
        loss_mse = self.loss_fn_mse(y_hat, labels)
        
        # Mean Absolute Error (MAE) loss
        loss_mae = self.loss_fn_mae(y_hat, labels)
        
        # Root Mean Squared Error (RMSE)
        loss_rmse = torch.sqrt(loss_mse)
        
        self.log('train_mse', loss_mse, on_epoch=True)
        self.log('train_mae', loss_mae, on_epoch=True)
        self.log('train_rmse', loss_rmse, on_epoch=True)
        
        return loss_mse

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        y_hat = self(inputs)
        # print('val', inputs.shape, labels.shape, y_hat.shape)

        # Mean Squared Error (MSE) loss
        loss_mse = self.loss_fn_mse(y_hat, labels)
        
        # Mean Absolute Error (MAE) loss
        loss_mae = self.loss_fn_mae(y_hat, labels)
        
        # Root Mean Squared Error (RMSE)
        loss_rmse = torch.sqrt(loss_mse)
        
        self.log('val_loss', loss_mse, on_epoch=True)
        self.log('val_mae', loss_mae, on_epoch=True)
        self.log('val_rmse', loss_rmse, on_epoch=True)
        
        return loss_mse

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.Tensor(self.X[index]), torch.Tensor([self.y[index]])