import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self,
                 sequence_len: int,
                 dimension_num: int,
                 activation: nn.Module) -> None:
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
