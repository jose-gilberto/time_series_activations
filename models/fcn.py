import torch
import torch.nn as nn


class FCN(nn.Module):

    def __init__(self, dimension_num: int, activation: nn.Module) -> None:
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
            nn.BatchNorm1d(256),
            activation,
            nn.AvgPool1d(kernel_size=1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x