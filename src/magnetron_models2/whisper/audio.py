# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import math
from pathlib import Path
from typing import Any, Iterator

from magnetron import Tensor, dtype, context

SAMPLE_RATE: int = 16000
NUM_FFT: int = 400
HOP_LEN: int = 160
CHUNK_LEN: int = 30
NUM_SAMPLES: int = CHUNK_LEN * SAMPLE_RATE
NUM_FRAMES: int = NUM_SAMPLES // HOP_LEN
NUM_FREQS: int = NUM_FFT // 2 + 1
_HZ_PER_MEL = 200.0 / 3.0  # Slaney linear region - 66.67 Hz per mel
_LOG_START_HZ = 1000.0  # linear below and logarithmic above
_LOG_START_MEL = _LOG_START_HZ / _HZ_PER_MEL  # = 15 mel
_LOG_STEP = math.log(6.4) / 27.0  # 27 mels per 6.4x in frequency


def load_audio_and_resample(path: Path) -> Tensor:
    audio, sample_rate = Tensor.load_audio(path)
    audio = audio.mean(dim=0)
    if (
        sample_rate != SAMPLE_RATE
    ):  # Resample by treating audio as image, bilinear is just linear interpolation on 2 axes so this works but not the most precise
        num = audio.shape[0]
        target = round(num * SAMPLE_RATE / sample_rate)  # Todo: add proper audio resampler
        audio = audio.reshape(1, 1, 1, num).interpolate(size=(1, target), mode='bilinear', antialias=True).reshape(target)
    return audio


def _dft_basis(n_fft: int) -> tuple[Tensor, Tensor]:
    n_freqs = n_fft // 2 + 1
    n = Tensor.arange(0, n_fft, dtype=dtype.int64).reshape(-1, 1)
    k = Tensor.arange(0, n_freqs, dtype=dtype.int64).reshape(1, -1)
    phase = ((n * k) % n_fft).cast(dtype.float32) * (math.pi * 2 / n_fft)
    window = (0.5 - 0.5 * (Tensor.arange(0, n_fft, dtype=dtype.float32) * (math.pi * 2 / n_fft)).cos()).reshape(-1, 1)  # Periodic Hann window
    return window * phase.cos(), window * phase.sin()


class LogMelSpectogram:
    def __init__(self, mel_filters: Tensor, n_fft: int, hop_len: int, samples: int) -> None:
        n_mels, n_freqs = mel_filters.shape
        assert n_freqs == n_fft // 2 + 1, f'Invalid frequency bin count: {n_freqs}, requires {n_fft // 2 + 1}'
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.samples = samples
        self.n_frames = samples // hop_len
        self.filtersT = mel_filters.cast(dtype.float32).T.contiguous().transfer(context.get_default_device())  # (n_freqs, n_mels)
        self.cos_base, self.sin_base = _dft_basis(n_fft)  # (n_fft, n_freqs) each

    @classmethod
    def from_snapshot(cls, mel_filters: Tensor, pp_cfg: dict[str, Any]) -> LogMelSpectogram:
        return cls(mel_filters, pp_cfg['n_fft'], pp_cfg['hop_length'], pp_cfg['n_samples'])

    def __call__(self, chunk: Tensor) -> Tensor:
        num = chunk.shape[0]
        if num < self.samples:
            chunk = chunk.pad((0, self.samples - num))
        padded = chunk.pad((self.n_fft // 2, self.n_fft // 2), 'reflect').contiguous()
        frames = padded.strided_view((self.n_frames, self.n_fft), (self.hop_len, 1))
        re = frames @ self.cos_base
        im = frames @ self.sin_base
        pow = re**2 + im**2
        log_spec = (pow @ self.filtersT).clamp_min(1e-10).log10()
        log_spec = log_spec.max(log_spec.max() - 8)
        return ((log_spec + 4) / 4).T.contiguous()


def chunks(audio: Tensor) -> Iterator[tuple[float, Tensor]]:
    num = max(audio.numel, 1)
    for start in range(0, num, NUM_SAMPLES):
        yield start / SAMPLE_RATE, audio[start : min(start + NUM_SAMPLES, num)].contiguous()
