import torch
from torch import nn

class MLP(nn.Module):

    def __init__(self,
                 sequence_len: int,
                 dimension_num: int,
                 activation: nn.Module,
                 **kwargs) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        
        self.layers = nn.ModuleList([
            nn.Dropout(p=0.1),
            nn.Linear(in_features=sequence_len * dimension_num,
                      out_features=500),
            activation,
            
            nn.Dropout(p=0.2),
            nn.Linear(in_features=500, out_features=500),
            activation,
            
            nn.Dropout(p=0.2),
            nn.Linear(in_features=500, out_features=500),
            activation,
            
            nn.Dropout(p=0.3)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = self.flatten(x)
        for layer in self.layers:
            x_ = layer(x_)
        return x_

class MLPClassifier(MLP):
    def __init__(self, sequence_len: int, dimension_num: int, activation: nn.Module, num_classes: int, **kwargs) -> None:
        super().__init__(sequence_len, dimension_num, activation, **kwargs)
        self.num_classes = num_classes
        self.output_layer = nn.Linear(in_features=500, out_features=num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = super().forward(x)
        x_ = self.output_layer(x_)
        return x_

class MLPRegressor(MLP):
    def __init__(self, sequence_len: int, dimension_num: int, activation: nn.Module, output_size: int = 1, **kwargs) -> None:
        super().__init__(sequence_len, dimension_num, activation, **kwargs)
        self.output_layer = nn.Linear(in_features=500, out_features=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = super().forward(x)
        return self.output_layer(x_)