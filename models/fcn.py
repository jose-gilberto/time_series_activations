import torch
import torch.nn as nn

class FCN(nn.Module):
  def __init__(self, input_shape, nb_classes = 1, n_feature_maps = 64, act_fn = nn.ReLU()):
    super().__init__()

    # Bloco 1
    self.conv1 = nn.Conv1d(in_channels=input_shape[0],
                           out_channels=128,
                           kernel_size=8,
                           padding='same')
    self.bn1 = nn.BatchNorm1d(128)
    self.act1 = act_fn

    # Bloco 2
    self.conv2 = nn.Conv1d(in_channels=128,
                           out_channels=256,
                           kernel_size=5,
                           padding='same')
    self.bn2 = nn.BatchNorm1d(256)
    self.act2 = act_fn

    # Bloco 3
    self.conv3 = nn.Conv1d(in_channels=256,
                           out_channels=128,
                           kernel_size=3,
                           padding='same')
    self.bn3 = nn.BatchNorm1d(128)
    self.act3 = act_fn

    # Saida
    self.avg_pool = nn.AvgPool1d(kernel_size=1)
    self.flat = nn.Flatten()
    self.linear = nn.Linear(128 * input_shape[1], nb_classes)


  def forward(self, x):
    # Bloco 1
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)

    # Bloco 2
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.act2(x)

    # Bloco 3
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.act3(x)

    # Saida
    x = self.avg_pool(x)
    x = self.flat(x)
    x = self.linear(x)
    return x