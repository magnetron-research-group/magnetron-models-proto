# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from dataclasses import dataclass
from magnetron_models2.common import ConfigBase


@dataclass
class Config(ConfigBase):
    repo_id: str = 'openai/whisper-base'
    vocab_size: int = 51865
    num_mel_bins: int = 80
    d_model: int = 512
    encoder_layers: int = 6
    encoder_attention_heads: int = 8
    encoder_ffn_dim: int = 2048
    decoder_layers: int = 6
    decoder_attention_heads: int = 8
    decoder_ffn_dim: int = 2048
    max_source_positions: int = 1500
    max_target_positions: int = 448
