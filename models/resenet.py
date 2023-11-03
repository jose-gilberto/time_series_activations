import torch
import torch.nn as nn

class Classifier_RESNET(nn.Module):
    def __init__(self, input_shape, nb_classes = 1, n_feature_maps = 64, act_fn = nn.ReLU()):
        super(Classifier_RESNET, self).__init__()

        # BLOCK 1
        self.conv1 = nn.Conv1d(input_shape[0], n_feature_maps, kernel_size=8, padding='same')
        self.batchnorm1 = nn.BatchNorm1d(n_feature_maps)
        self.act_fn = act_fn

        self.conv2 = nn.Conv1d(n_feature_maps, n_feature_maps, kernel_size=5, padding='same')
        self.batchnorm2 = nn.BatchNorm1d(n_feature_maps)

        self.conv3 = nn.Conv1d(n_feature_maps, n_feature_maps, kernel_size=3, padding='same')
        self.batchnorm3 = nn.BatchNorm1d(n_feature_maps)

        self.conv_shortcut = nn.Conv1d(input_shape[0], n_feature_maps, kernel_size=1, padding='same')

        # BLOCK 2
        self.conv4 = nn.Conv1d(n_feature_maps, n_feature_maps * 2, kernel_size=8, padding='same')
        self.batchnorm4 = nn.BatchNorm1d(n_feature_maps * 2)

        self.conv5 = nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, kernel_size=5, padding='same')
        self.batchnorm5 = nn.BatchNorm1d(n_feature_maps * 2)

        self.conv6 = nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, kernel_size=3, padding='same')
        self.batchnorm6 = nn.BatchNorm1d(n_feature_maps * 2)

        self.conv_shortcut2 = nn.Conv1d(n_feature_maps, n_feature_maps * 2, kernel_size=1, padding='same')

        # BLOCK 3
        self.conv7 = nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, kernel_size=8, padding='same')
        self.batchnorm7 = nn.BatchNorm1d(n_feature_maps * 2)

        self.conv8 = nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, kernel_size=5, padding='same')
        self.batchnorm8 = nn.BatchNorm1d(n_feature_maps * 2)

        self.conv9 = nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, kernel_size=3, padding='same')
        self.batchnorm9 = nn.BatchNorm1d(n_feature_maps * 2)

        # FINAL
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_feature_maps * 2, nb_classes)

    def forward(self, x):
        # BLOCK 1
        conv_x = self.conv1(x)
        conv_x = self.batchnorm1(conv_x)
        conv_x = self.act_fn(conv_x)

        conv_y = self.conv2(conv_x)
        conv_y = self.batchnorm2(conv_y)
        conv_y = self.act_fn(conv_y)

        conv_z = self.conv3(conv_y)
        conv_z = self.batchnorm3(conv_z)

        # expand channels for the sum
        shortcut_y = self.conv_shortcut(x)
        shortcut_y = self.batchnorm1(shortcut_y)

        output_block_1 = shortcut_y + conv_z
        output_block_1 = self.act_fn(output_block_1)

        # BLOCK 2
        conv_x = self.conv4(output_block_1)
        conv_x = self.batchnorm4(conv_x)
        conv_x = self.act_fn(conv_x)

        conv_y = self.conv5(conv_x)
        conv_y = self.batchnorm5(conv_y)
        conv_y = self.act_fn(conv_y)

        conv_z = self.conv6(conv_y)
        conv_z = self.batchnorm6(conv_z)

        # expand channels for the sum
        shortcut_y = self.conv_shortcut2(output_block_1)
        shortcut_y = self.batchnorm4(shortcut_y)

        output_block_2 = shortcut_y + conv_z
        output_block_2 = self.act_fn(output_block_2)

        # BLOCK 3
        conv_x = self.conv7(output_block_2)
        conv_x = self.batchnorm7(conv_x)
        conv_x = self.act_fn(conv_x)

        conv_y = self.conv8(conv_x)
        conv_y = self.batchnorm8(conv_y)
        conv_y = self.act_fn(conv_y)

        conv_z = self.conv9(conv_y)
        conv_z = self.batchnorm9(conv_z)

        shortcut_y = self.batchnorm7(output_block_2)

        output_block_3 = shortcut_y + conv_z
        output_block_3 = self.act_fn(output_block_3)

        # FINAL
        gap_layer = self.global_avg_pooling(output_block_3)
        gap_layer = gap_layer.squeeze()
        output_layer = self.fc(gap_layer)

        return output_layer