import torch
import torch.nn as nn

class GAP1d(nn.Module):

    def __init__(self, output_size: int = 1) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = nn.Flatten()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.gap(x))


class ResNetBlock(nn.Module):

    def __init__(self, dimension_num, out_channels, activation: nn.Module, **kwargs) -> None:
        super().__init__()

        # Convolutios by kernel num
        self.conv_8 = nn.Conv1d(dimension_num,
                                out_channels,
                                kernel_size=8,
                                padding='same')
        self.conv_5 = nn.Conv1d(out_channels,
                                out_channels,
                                kernel_size=5,
                                padding='same')
        self.conv_3 = nn.Conv1d(out_channels,
                                out_channels,
                                kernel_size=8,
                                padding='same')

        self.conv_shortcut = nn.Conv1d(dimension_num,
                                       out_channels,
                                       kernel_size=1,
                                       padding='same')

        self.bn_8 = nn.BatchNorm1d(out_channels)
        self.bn_5 = nn.BatchNorm1d(out_channels)
        self.bn_3 = nn.BatchNorm1d(out_channels)

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First convolution with kernel 8
        conv_x = self.conv_8(x)
        conv_x = self.bn_8(conv_x)
        conv_x = self.activation(conv_x)

        # Second convolution with kernel 5
        conv_y = self.conv_5(conv_x)
        conv_y = self.bn_5(conv_y)
        conv_y = self.activation(conv_y)

        # Third convolution with kernel 3
        conv_z = self.conv_3(conv_y)
        conv_z = self.bn_3(conv_z)

        # Expand channels for the sum with shortcut
        shortcut_ = self.conv_shortcut(x)
        shortcut_ = self.bn_8(shortcut_)

        # Prepare the output summing the shortcut
        out = shortcut_ + conv_z
        out = self.activation(out)
        return out


class ResNet(nn.Module):

    def __init__(self, dimension_num: int, out_channels: int, activation: nn.Module, **kwargs) -> None:
        super().__init__()

        self.block_1 = ResNetBlock(dimension_num, out_channels, activation)
        self.block_2 = ResNetBlock(out_channels, out_channels * 2, activation)
        self.block_3 = ResNetBlock(out_channels * 2, out_channels * 2, activation)

        self.global_avg_pooling = GAP1d()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_1 = self.block_1(x)
        out_2 = self.block_2(out_1)
        out_3 = self.block_3(out_2)

        gap_ = self.global_avg_pooling(out_3)
        # gap_ = gap_layer.squeeze()
        return gap_
    

class ResNetClassifier(ResNet):
    def __init__(self, dimension_num: int, out_channels: int, activation: nn.Module, num_classes: int, **kwargs) -> None:
        super().__init__(dimension_num, out_channels, activation, **kwargs)
        self.num_classes = num_classes
        self.output_layer = nn.Linear(out_channels * 2, num_classes)

    def forward(self, x):
        x_ = super().forward(x)
        x_ = self.output_layer(x_)
        return x_


class ResNetRegressor(ResNet):
    def __init__(self, dimension_num: int, out_channels: int, activation: nn.Module, output_size: int = 1, **kwargs) -> None:
        super().__init__(dimension_num, out_channels, activation, **kwargs)
        self.output_layer = nn.Linear(out_channels * 2, output_size)

    def forward(self, x):
        x_ = super().forward(x)
        return self.output_layer(x_)