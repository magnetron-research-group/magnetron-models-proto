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
from typing import Iterator

from magnetron import Tensor, nn, dtype, context
from magnetron.snapshot import deserialize
from magnetron_models2.common import ModelBase, ConfigBase
from magnetron_models2.kvcache import KVLayerCache
from magnetron_models2.whisper.config import Config


class Attn(nn.Module):
    @nn.layer_init(weight=nn.init.EmptyInitStrategy(), bias=nn.init.EmptyInitStrategy())
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        return x.reshape(B, T, self.heads, self.head_dim).transpose(1, 2)

    def project_kv(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self._split_heads(self.k_proj(x)), self._split_heads(self.v_proj(x))

    def attend(self, x: Tensor, k: Tensor, v: Tensor, causal: bool = False) -> Tensor:
        B, T, C = x.shape
        q = self._split_heads(self.q_proj(x))
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if causal and T > 1:
            S = k.shape[2]
            k_pos = Tensor.arange(S).reshape(1, -1)
            q_pos = Tensor.arange(start=S - T, stop=S).reshape(-1, 1)
            mask = Tensor.where(k_pos <= q_pos, 0, -1e-4).cast(scores.dtype).reshape(1, 1, T, S)
            scores = scores + mask
        return self.out_proj((scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C))

    def forward(self, x: Tensor) -> Tensor:
        k, v = self.project_kv(x)
        return self.attend(x, k, v)


class Encoder(nn.Module):
    class Layer(nn.Module):
        @nn.layer_init(weight=nn.init.EmptyInitStrategy(), bias=nn.init.EmptyInitStrategy())
        def __init__(self, cfg: Config) -> None:
            super().__init__()
            self.cfg = cfg
            self.self_attn = Attn(cfg.d_model, cfg.encoder_attention_heads)
            self.self_attn_layer_norm = nn.LayerNorm(cfg.d_model)
            self.fc1 = nn.Linear(cfg.d_model, cfg.encoder_ffn_dim)
            self.fc2 = nn.Linear(cfg.encoder_ffn_dim, cfg.d_model)
            self.final_layer_norm = nn.LayerNorm(cfg.d_model)

        def forward(self, x: Tensor) -> Tensor:
            x = x + self.self_attn(self.self_attn_layer_norm(x))
            x = x + self.fc2(self.fc1(self.final_layer_norm(x)).gelu())
            return x

    @nn.layer_init(weight=nn.init.EmptyInitStrategy(), bias=nn.init.EmptyInitStrategy())
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.conv1 = nn.Conv1D(cfg.num_mel_bins, cfg.d_model, 3, padding=1)
        self.conv2 = nn.Conv1D(cfg.d_model, cfg.d_model, 3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(cfg.max_source_positions, cfg.d_model)
        self.layers = nn.ModuleList([self.Layer(cfg) for _ in range(cfg.encoder_layers)])
        self.layer_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, mel: Tensor) -> Tensor:
        x = self.conv1(mel).gelu()
        x = self.conv2(x).gelu()
        x = x.transpose(1, 2)
        x = x + self.embed_positions.weight[: x.shape[1]]
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class Decoder(nn.Module):
    class Layer(nn.Module):
        @nn.layer_init(weight=nn.init.EmptyInitStrategy(), bias=nn.init.EmptyInitStrategy())
        def __init__(self, cfg: Config) -> None:
            super().__init__()
            self.self_attn = Attn(cfg.d_model, cfg.decoder_attention_heads)
            self.self_attn_layer_norm = nn.LayerNorm(cfg.d_model)
            self.encoder_attn = Attn(cfg.d_model, cfg.decoder_attention_heads)
            self.encoder_attn_layer_norm = nn.LayerNorm(cfg.d_model)
            self.fc1 = nn.Linear(cfg.d_model, cfg.decoder_ffn_dim)
            self.fc2 = nn.Linear(cfg.decoder_ffn_dim, cfg.d_model)
            self.final_layer_norm = nn.LayerNorm(cfg.d_model)

        def forward(self, x: Tensor, cache: KVLayerCache, cross_kv: tuple[Tensor, Tensor]) -> Tensor:
            h = self.self_attn_layer_norm(x)
            k, v = cache.append(*self.self_attn.project_kv(h))
            x = x + self.self_attn.attend(h, k, v, causal=True)
            x = x + self.encoder_attn.attend(self.encoder_attn_layer_norm(x), *cross_kv)
            x = x + self.fc2(self.fc1(self.final_layer_norm(x)).gelu())
            return x

    @nn.layer_init(weight=nn.init.EmptyInitStrategy(), bias=nn.init.EmptyInitStrategy())
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_positions = nn.Embedding(cfg.max_target_positions, cfg.d_model)
        self.layers = nn.ModuleList([self.Layer(cfg) for _ in range(cfg.decoder_layers)])
        self.layer_norm = nn.LayerNorm(cfg.d_model)
        self.caches: list[KVLayerCache] = []
        self.cross_kv: list[tuple[Tensor, Tensor]] = []

    def reset(self, out_encoder: Tensor) -> None:
        head_dim = self.cfg.d_model // self.cfg.decoder_attention_heads
        self.caches = [KVLayerCache(1, self.cfg.decoder_attention_heads, self.cfg.max_target_positions, head_dim) for _ in self.layers]
        self.cross_kv = [layer.encoder_attn.project_kv(out_encoder) for layer in self.layers]

    @property
    def pos(self) -> int:
        return self.caches[0].pos

    def forward(self, tokens: Tensor) -> Tensor:
        pos = self.pos
        x = self.embed_tokens(tokens) + self.embed_positions.weight[pos : pos + tokens.shape[1]]
        for layer, cache, cross_kv in zip(self.layers, self.caches, self.cross_kv, strict=True):
            x = layer(x, cache, cross_kv)
        x = self.layer_norm(x[:, -1, :])
        return (x @ self.embed_tokens.weight.T).reshape(-1)


class WhisperModel(ModelBase):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg, is_trainable=False)
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    @staticmethod
    def from_snapshot(file: Path) -> tuple[dict[str, Tensor], WhisperModel]:
        tensors, meta = deserialize(file)
        if meta.get('arch') != 'whisper':
            raise RuntimeError(f'{file} holds a {meta.get("arch") or "nameless"} model, not whisper')
        context.set_default_dtype(getattr(dtype, meta['dtype']))
        model = WhisperModel(ConfigBase.decode(Config, meta['mod_cfg']))
        model.load_tensors(tensors, meta)
        return tensors, model

    def encode(self, mel: Tensor) -> Tensor:
        return self.encoder(mel)

    def detect_language(self, out_encoder: Tensor, sot: int, lang_ids: Tensor) -> Tensor:
        self.decoder.reset(out_encoder)
        logits = self.decoder(Tensor([[sot]], dtype=dtype.int64))
        return lang_ids[logits[lang_ids].argmax(dim=0)]

    def generate(self, out_encoder: Tensor, prompt: Tensor, eos: int, suppress: Tensor, begin_supress: Tensor, max_tokens: int) -> Tensor:
        self.decoder.reset(out_encoder)
        logits = self.decoder(prompt)
        max_toks = min(max_tokens, self.cfg.max_target_positions - prompt.numel)
        toks: list[Tensor] = []
        for i in range(max_toks):
            logits = logits + (suppress + begin_supress if i == 0 else suppress)
            tok = logits.argmax(dim=0)
            if tok.item() == eos:
                break
            toks.append(tok)
            logits = self.decoder(tok.reshape(1, 1))
        return Tensor.stack(toks)
