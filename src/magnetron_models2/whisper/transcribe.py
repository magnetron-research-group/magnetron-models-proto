# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+
from pathlib import Path
from typing import Any

from magnetron import context, dtype, Tensor
from magnetron_models2.whisper.audio import LogMelSpectogram, load_audio_and_resample, SAMPLE_RATE, chunks
from magnetron_models2.whisper.config import Config
from magnetron_models2.whisper.model import WhisperModel
from tokenizers import Tokenizer

context.set_default_device('cpu')


class Transcriber:
    def __init__(self, snapshot: Path) -> None:
        context.stop_grad_recorder()
        tensors, model = WhisperModel.from_snapshot(snapshot)
        self.model = model
        meta: dict[str, Any] = model.snapshot_metadata
        gen: dict[str, Any] = meta.get('gen_cfg') or {}
        hf: dict[str, Any] = meta['hf_cfg']
        self.tokenizer = Tokenizer.from_str(meta['tokenizer_json'])
        self.mel = LogMelSpectogram.from_snapshot(tensors['mel_filters'], meta['pp_cfg'])
        self.lang2id: dict[str, int] = gen['lang_to_id']
        self.lang2idt: Tensor = Tensor(list(self.lang2id.values()), dtype=dtype.int64)
        self.task_to_id: dict[str, int] = gen['task_to_id']
        self.sot: int = gen.get('decoder_start_token_id', hf['decoder_start_token_id'])
        self.eos: int = gen.get('eos_token_id', hf['eos_token_id'])
        self.no_timestamps: int = gen.get('no_timestamps_token_id', self.tokenizer.token_to_id('<|notimestamps|>'))
        self.suppress = self._bias(gen.get('suppress_tokens', hf.get('suppress_tokens')) or [])
        self.begin_suppress = self._bias(gen.get('begin_suppress_tokens', hf.get('begin_suppress_tokens')) or [])

    def _bias(self, token_ids: list[int]) -> Tensor:
        vocab = self.model.cfg.vocab_size
        mask = Tensor(token_ids, dtype=dtype.int64).bincount(minlength=vocab) > 0
        return Tensor.zeros(vocab, dtype=context.get_default_dtype()).masked_fill(mask, float('-inf'))

    def _construct_prompt(self, lang: str) -> Tensor:
        return Tensor([self.sot, self.lang2id[lang], self.task_to_id['transcribe'], self.no_timestamps])[None, ...]

    def transcribe(self, audio: Tensor, max_tok: int = 224) -> str:
        assert audio.rank == 1 and audio.numel > 0
        dur = audio.numel / SAMPLE_RATE
        print(f'{dur:.1f}s of audio')
        parts: list[str] = []
        prompt: Tensor | None = None
        for i, (offset, chunk) in enumerate(chunks(audio)):
            mel = self.mel(chunk).cast(context.get_default_dtype())
            out_encoder = self.model.encoder(mel.reshape(1, *mel.shape))
            if i == 0:  # detect language and init prompt
                lang_id: int = self.model.detect_language(out_encoder, self.sot, self.lang2idt).item()
                lang_tok = next(k for k, v in self.lang2id.items() if v == lang_id)
                prompt = self._construct_prompt(lang_tok)
            assert prompt is not None
            tokens: Tensor = self.model.generate(out_encoder, prompt, self.eos, self.suppress, self.begin_suppress, max_tokens=max_tok)
            parts.append(self.tokenizer.decode(tokens.tolist(), skip_special_tokens=True).strip())
        return ' '.join(part for part in parts if part)

    def transcribe_file(self, path: Path, max_tok: int = 224) -> str:
        return self.transcribe(load_audio_and_resample(path), max_tok)


transcriber = Transcriber('/home/mario/Desktop/whisper-base-bf16.mag')
print(transcriber.transcribe_file('speech.wav'))
