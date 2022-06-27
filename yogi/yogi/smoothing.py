import numpy as np
from scipy.interpolate import splev, splrep
import torch
import torch.nn as nn
import torch.nn.functional as F


def reach2tensor(reach):
    return torch.FloatTensor(reach).view(1, reach.shape[0], -1)


def mask2tensor(mask):
    return torch.LongTensor(mask).view(1, -1)


def detect_anomalies(reach, model, block_length, normalization_factor=20,
                     threshold=0.2):

    if (reach.shape[1] < block_length):
        print('Warning: trajetory is shorter than cnn block length')
        return np.zeros((reach.shape[1],), dtype=np.bool)

    score_sums = np.zeros(reach.shape[1])
    counts = np.zeros(reach.shape[1])
    with torch.no_grad():
        model.eval()
        for offset in range(reach.shape[1] - block_length + 1):
            reach_chunk = reach[:, offset:(offset + block_length)]
            inputs = reach2tensor(reach_chunk / normalization_factor)
            label_scores = model(inputs)
            anomaly_mask = np.exp(np.array(label_scores[0, 1, :])) > 0.001
            score_sums[offset:(offset + block_length)] += anomaly_mask
            counts[offset:(offset + block_length)] += 1.0
    anomalies = (score_sums / counts) > threshold

    # HACK: clip the ends, where detection is not reliable
    anomalies[:10] = False
    anomalies[-10:] = False

    return anomalies


def fill_anomalies_1d(reach, anomalies, smoothing_factor=0.005):
    idxs = range(0, len(anomalies))

    x = np.array(idxs)[~anomalies]
    y = np.array(reach[~anomalies])

    spl = splrep(x, y, s=smoothing_factor)
    x_interp = np.arange(0, len(anomalies))
    y_interp = splev(x_interp, spl)

    return y_interp


def fill_anomalies(reach, anomalies, smoothing_factor=0.005):
    x_interp = fill_anomalies_1d(reach[0, :], anomalies,
                                 smoothing_factor=smoothing_factor)
    y_interp = fill_anomalies_1d(reach[1, :], anomalies,
                                 smoothing_factor=smoothing_factor)
    reach_interp = np.array((x_interp, y_interp))
    return reach_interp


def smooth_interp(reach, smoothing_factor=0.005):
    x = np.array(range(0, reach.shape[1]))
    y = np.array(reach[0, :])

    spl = splrep(x, y, s=smoothing_factor)
    x_interp = range(0, reach.shape[1])
    y_interp = splev(x_interp, spl)

    reach_interp = np.array((x_interp, y_interp))

    return reach_interp


def nn_smooth(reach, model, anomaly_thresh=0.2, smoothing_factor=0.005,
              block_length=128, normalization_factor=20):
    anomalies = detect_anomalies(reach, model, block_length,
                                 normalization_factor=normalization_factor,
                                 threshold=anomaly_thresh)
    smooth_reach = fill_anomalies(reach, anomalies, smoothing_factor=smoothing_factor)
    return smooth_reach


INPUT_CHANNELS = 2
N_LABELS = 2
KERNEL_SIZE = 3
PADDING = int((KERNEL_SIZE - 1) / 2)


class UNetBlock(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(UNetBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_channels,
                               out_channels=output_channels,
                               kernel_size=KERNEL_SIZE,
                               padding_mode='zeros',
                               padding=PADDING)

        self.bn1 = nn.BatchNorm1d(num_features=output_channels)

        self.conv2 = nn.Conv1d(in_channels=output_channels,
                               out_channels=output_channels,
                               kernel_size=KERNEL_SIZE,
                               padding_mode='zeros',
                               padding=PADDING)

        self.bn2 = nn.BatchNorm1d(num_features=output_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


def make_unet_block(input_channels, output_channels):
    block = UNetBlock(input_channels=input_channels,
                      output_channels=output_channels)
    return block


def make_upsample_conv(input_channels, output_channels):
    conv = nn.Conv1d(in_channels=input_channels,
                     out_channels=output_channels,
                     kernel_size=KERNEL_SIZE,
                     padding_mode='zeros',
                     padding=PADDING)
    return conv


class UNet(nn.Module):

    def __init__(self, scale_factor):
        super(UNet, self).__init__()

        self.scale_factor = scale_factor

        # encoders

        self.num_encoders = 5
        encoder_blocks = []

        input_channels = 2
        output_channels = 16

        for i in range(self.num_encoders):
            block = make_unet_block(input_channels, output_channels)
            encoder_blocks.append(block)
            input_channels = output_channels
            output_channels = output_channels * 2

        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        # decoders

        self.num_decoders = 4
        decoder_blocks = []
        upsample_convs = []

        output_channels = int(input_channels / 2)

        for i in range(self.num_decoders):
            upsample_conv = make_upsample_conv(input_channels=input_channels,
                                               output_channels=output_channels)
            upsample_convs.append(upsample_conv)
            block = make_unet_block(input_channels, output_channels)
            decoder_blocks.append(block)
            input_channels = output_channels
            output_channels = int(output_channels / 2)

        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.upsample_convs = nn.ModuleList(upsample_convs)

        # output layer

        output_channels = N_LABELS

        self.output_conv = nn.Conv1d(in_channels=input_channels,
                                     out_channels=N_LABELS,
                                     kernel_size=KERNEL_SIZE,
                                     padding_mode='zeros',
                                     padding=PADDING)

    def forward(self, sequence):
        assert(sequence.shape[1])

        x = sequence

        skip_variables = []

        for i in range(self.num_encoders - 1):
            x = self.encoder_blocks[i](x)
            skip_variables.append(x)
            x = F.max_pool1d(x, kernel_size=self.scale_factor)

        x = self.encoder_blocks[-1](x)

        for i in range(self.num_decoders):
            x = nn.Upsample(scale_factor=self.scale_factor)(x)
            x = self.upsample_convs[i](x)
            x = torch.cat([x, skip_variables[-1 - i]], axis=1)
            x = self.decoder_blocks[i](x)

        x = self.output_conv(x)
        x = F.log_softmax(x, dim=1)

        return x
