import torch
import torch.nn as nn

class GAP1d(nn.Module):

    def __init__(self, output_size: int = 1) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = nn.Flatten()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.gap(x))


class FCN(nn.Module):

    def __init__(self,
                 dimension_num: int, 
                 activation: nn.Module,
                 **kwargs) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Conv1d(in_channels=dimension_num,
                      out_channels=128,
                      kernel_size=8,
                      padding='same'),
            nn.BatchNorm1d(128),
            activation,
            nn.Conv1d(in_channels=128,
                      out_channels=256,
                      kernel_size=5,
                      padding='same'),
            nn.BatchNorm1d(256),
            activation,
            nn.Conv1d(in_channels=256,
                      out_channels=128,
                      kernel_size=3,
                      padding='same'),
            nn.BatchNorm1d(128),
            activation,
            GAP1d()
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
class FCNClassifier(FCN):
    def __init__(self, dimension_num: int, activation: nn.Module, num_classes: int, **kwargs) -> None:
        super().__init__(dimension_num, activation, **kwargs)
        self.output_layer = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = super().forward(x)
        x_ = self.output_layer(x_)
        return nn.functional.softmax(x_, dim=1)


class FCNRegressor(FCN):
    def __init__(self, dimension_num: int, activation: nn.Module, output_size: int = 1, **kwargs) -> None:
        super().__init__(dimension_num, activation, **kwargs)
        self.output_layer = nn.Linear(in_features=128, out_features=output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = super().forward(x)
        return self.output_layer(x_)