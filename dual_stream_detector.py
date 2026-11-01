import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_linear_filterbanks(n_filters=20, n_fft=512, sample_rate=16000,
                           low_freq=0.0, high_freq=8000.0):
    num_bins = n_fft // 2 + 1
    fft_freqs = torch.linspace(0, sample_rate / 2.0, num_bins)
    filter_freqs = torch.linspace(low_freq, high_freq, n_filters + 2)
    fbank = torch.zeros(n_filters, num_bins)
    for i in range(n_filters):
        left, center, right = filter_freqs[i:i+3]
        mask = (fft_freqs >= left) & (fft_freqs <= center)
        if center > left:
            fbank[i, mask] = (fft_freqs[mask] - left) / (center - left)
        mask = (fft_freqs >= center) & (fft_freqs <= right)
        if right > center:
            fbank[i, mask] = (right - fft_freqs[mask]) / (right - center)
    return fbank


def get_mel_filterbanks(n_mels=128, n_fft=1024, sample_rate=16000,
                        low_freq=0.0, high_freq=8000.0):
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mel_points = np.linspace(hz_to_mel(low_freq), hz_to_mel(high_freq), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    fft_freqs = torch.linspace(0, sample_rate / 2.0, n_fft // 2 + 1)
    fbank = torch.zeros(n_mels, n_fft // 2 + 1)
    for i in range(n_mels):
        left, center, right = hz_points[i:i+3]
        mask = (fft_freqs >= left) & (fft_freqs <= center)
        if center > left:
            fbank[i, mask] = (fft_freqs[mask] - left) / (center - left)
        mask = (fft_freqs >= center) & (fft_freqs <= right)
        if right > center:
            fbank[i, mask] = (right - fft_freqs[mask]) / (right - center)
    return fbank


def compute_deltas(spec, width=9):
    _, _, n_frames = spec.shape
    half = width // 2
    denom = 2 * sum(i**2 for i in range(1, half + 1))
    padded = F.pad(spec, (half, half), mode="replicate")
    delta = torch.zeros_like(spec)
    for i in range(1, half + 1):
        delta += i * (
            padded[:, :, half+i:half+i+n_frames]
            - padded[:, :, half-i:half-i+n_frames]
        )
    return delta / denom


class DualStreamFeatureExtractor(nn.Module):
    def __init__(self, sample_rate=16000, n_lfcc=20, n_mels=128, target_frames=126):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_lfcc = n_lfcc
        self.n_mels = n_mels
        self.target_frames = target_frames

        self.register_buffer("lfcc_window", torch.hann_window(400))
        self.register_buffer("linear_fbank",
                             get_linear_filterbanks(n_lfcc, 512, sample_rate))
        n = torch.arange(n_lfcc).unsqueeze(1)
        k = torch.arange(n_lfcc).unsqueeze(0)
        dct = torch.cos(math.pi * n * (k + 0.5) / n_lfcc) * math.sqrt(2.0 / n_lfcc)
        dct[0, :] *= 1.0 / math.sqrt(2.0)
        self.register_buffer("dct_matrix", dct)

        self.register_buffer("mel_window", torch.hann_window(640))
        self.register_buffer("mel_fbank",
                             get_mel_filterbanks(n_mels, 1024, sample_rate))

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        stft = torch.stft(
            x, n_fft=512, hop_length=160, win_length=400,
            window=self.lfcc_window, return_complex=True, center=True,
            pad_mode="reflect"
        )
        magnitude = torch.abs(stft)
        energy = torch.matmul(self.linear_fbank, magnitude)
        log_energy = torch.log(torch.clamp(energy, min=1e-8))
        cepstral = torch.matmul(self.dct_matrix, log_energy)
        delta1 = compute_deltas(cepstral)
        delta2 = compute_deltas(delta1)
        stacked = torch.cat([cepstral, delta1, delta2], dim=1)
        lfcc = torch.cat([
            stacked.mean(dim=2),
            stacked.std(dim=2, unbiased=False)
        ], dim=1)

        stft_mel = torch.stft(
            x, n_fft=1024, hop_length=160, win_length=640,
            window=self.mel_window, return_complex=True, center=True,
            pad_mode="reflect"
        )
        power = torch.abs(stft_mel) ** 2
        mel_energy = torch.matmul(self.mel_fbank, power)
        log_mel = torch.log1p(mel_energy)
        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True, unbiased=False) + 1e-6
        mel = ((log_mel - mean) / std).unsqueeze(1)
        if mel.shape[-1] != self.target_frames:
            mel = F.adaptive_avg_pool2d(mel, (self.n_mels, self.target_frames))
        return lfcc, mel


class LFCC_MLP(nn.Module):
    def __init__(self, in_dim=120, hidden_dim=64, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.block(x)


class Lightweight4BlockCNN(nn.Module):
    def __init__(self, in_channels=1, embed_dim=128):
        super().__init__()
        self.b1 = ConvBlock(in_channels, 16)
        self.b2 = ConvBlock(16, 32)
        self.b3 = ConvBlock(32, 64)
        self.b4 = ConvBlock(64, embed_dim)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        return self.dropout(self.gap(x).flatten(1))


class DualStreamDetector(nn.Module):
    """Fuses LFCC statistical features and Log-Mel time-frequency features."""
    def __init__(self):
        super().__init__()
        self.stream_a = LFCC_MLP(in_dim=120, hidden_dim=64, embed_dim=32)
        self.stream_b = Lightweight4BlockCNN(in_channels=1, embed_dim=128)
        self.fusion = nn.Sequential(
            nn.Linear(32 + 128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, lfcc, mel):
        a = self.stream_a(lfcc)
        b = self.stream_b(mel)
        return self.fusion(torch.cat([a, b], dim=1))
