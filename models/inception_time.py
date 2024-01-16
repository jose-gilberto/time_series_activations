import torch
from torch import nn
import pytorch_lightning as pl
from typing import Tuple


# Helper to shortcut for residual blocks, don't do anything
def noop(x: torch.Tensor) -> torch.Tensor:
    return x

class InceptionModule(nn.Module):
    def __init__(self, 
                 dimension_num: int, 
                 out_channels: int, 
                 activation: nn.Module,
                 kernel_size: Tuple[int, int, int] = [7, 5, 3],  # type: ignore
                 bottleneck: bool = True) -> None:
        """ Inception Module to apply parallel convolution on time series.

        Args:
            dimension_num (int): Number of dimensions.
            out_channels (int): Number of output channels.
            kernel_size (Tuple[int, int, int], optional): Number of kernels. Defaults to [7, 5, 3].
            bottleneck (bool, optional): If apply a bottleneck layer to convert the number of channels to the correct shape. Defaults to True.
        """
        super().__init__()
        
        self.kernel_sizes = kernel_size
        # Only apply bottleneck if the input channels number is bigger than 1
        bottleneck = bottleneck if dimension_num > 1 else False
        self.bottleneck = nn.Conv1d(dimension_num, out_channels, kernel_size=1, bias=False, padding='same') if bottleneck else noop
        # Calculate and apply convolutions for each kernel size
        self.convolutions = nn.ModuleList([
            nn.Conv1d(out_channels if bottleneck else dimension_num, out_channels, kernel_size=k, padding='same', bias=False) for k in self.kernel_sizes
        ])
        # Max Convolutional Pooling layer
        self.maxconv = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(dimension_num, out_channels, 1, bias=False, padding='same')])
        self.batchnorm = nn.BatchNorm1d(out_channels * 4)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = x
        x = self.bottleneck(x)
        # Conv1, Conv2, Conv3, MaxConv
        x = torch.cat([conv(x) for conv in self.convolutions] + [self.maxconv(x_)], dim=1)
        return self.activation(x)


class InceptionBlock(nn.Module):
    def __init__(self, dimension_num: int, out_channels: int, activation, residual: bool = True, depth: int = 6) -> None:
        super().__init__()
        self.residual = residual
        self.depth = depth
        self.activation = activation
        
        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        for d in range(depth):
            # Build each inception module
            self.inception.append(InceptionModule(
                dimension_num=(dimension_num if d == 0 else out_channels * 4), out_channels=out_channels, activation=activation
            ))
            if self.residual and d % 3 == 2:
                c_in, c_out = dimension_num if d == 2 else out_channels * 4, out_channels * 4
                self.shortcut.append(
                    nn.BatchNorm1d(c_in) if c_in == c_out else nn.Sequential(*[nn.Conv1d(c_in, c_out, kernel_size=1, padding='same'), nn.BatchNorm1d(c_out)])
                )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)
            if self.residual and d % 3 == 2:
               res = x = self.activation(x + self.shortcut[d // 3](res))
        return x


class InceptionTime(nn.Module):
    def __init__(self, dimension_num: int, 
                 hidden_channels: int, 
                 num_classes: int, 
                 activation: nn.Module, 
                 **kwargs) -> None:
        super().__init__()
        self.inception_block = InceptionBlock(dimension_num, hidden_channels, activation)
        self.fc = nn.Linear(hidden_channels * 4, num_classes)
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inception_block(x)
        x = torch.mean(x, dim=-1)
        return self.activation(self.fc(x))
    
class InceptionTimeClassifier(InceptionTime):
    def __init__(self, dimension_num: int, hidden_channels: int, num_classes: int, activation: nn.Module,**kwargs) -> None:
        super().__init__(dimension_num, hidden_channels, num_classes, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

class InceptionTimeRegressor(InceptionTime):
    def __init__(self, dimension_num: int, hidden_channels: int, activation: nn.Module, num_classes: int = 1, **kwargs) -> None:
        super().__init__(dimension_num, hidden_channels, num_classes, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)
