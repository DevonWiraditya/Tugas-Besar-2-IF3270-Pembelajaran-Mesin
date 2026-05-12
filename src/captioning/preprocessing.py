from __future__ import annotations

from collections import Counter
import re
from typing import Iterable

import numpy as np

from src.captioning.config import CaptioningConfig
from src.captioning.data import CaptionDataset, Vocabulary
from src.utils.io import write_json

_PUNCTUATION_RE = re.compile(r'[^a-z0-9\s]')
_MULTISPACE_RE = re.compile(r'\s+')


def clean_caption_text(text: str, config: CaptioningConfig) -> str:
    value = text.strip()
    if config.text.lowercase:
        value = value.lower()
    if config.text.strip_punctuation:
        value = _PUNCTUATION_RE.sub(' ', value)
    return _MULTISPACE_RE.sub(' ', value).strip()


def tokenize_caption(text: str) -> list[str]:
    return [token for token in text.split(' ') if token]


def build_vocabulary(dataset: CaptionDataset, config: CaptioningConfig) -> Vocabulary:
    counts: Counter[str] = Counter()
    for record in dataset.train.records:
        counts.update(tokenize_caption(clean_caption_text(record.raw_caption, config)))
    tokens = [config.text.pad_token, config.text.start_token, config.text.end_token, config.text.oov_token]
    for token, count in sorted(counts.items()):
        if count >= config.text.min_word_frequency and token not in tokens:
            tokens.append(token)
    token_to_id = {token: index for index, token in enumerate(tokens)}
    id_to_token = {index: token for token, index in token_to_id.items()}
    return Vocabulary(token_to_id=token_to_id, id_to_token=id_to_token, pad_id=token_to_id[config.text.pad_token], start_id=token_to_id[config.text.start_token], end_id=token_to_id[config.text.end_token], oov_id=token_to_id[config.text.oov_token])


def encode_caption(text: str, vocabulary: Vocabulary, config: CaptioningConfig) -> list[int]:
    token_ids = [vocabulary.start_id]
    for token in tokenize_caption(clean_caption_text(text, config)):
        token_ids.append(vocabulary.token_to_id.get(token, vocabulary.oov_id))
    token_ids.append(vocabulary.end_id)
    return token_ids[: config.text.max_caption_length]


def pad_encoded_sequences(sequences: Iterable[list[int]], vocabulary: Vocabulary, max_length: int) -> np.ndarray:
    rows = []
    for sequence in sequences:
        padded = sequence[:max_length]
        padded = padded + [vocabulary.pad_id] * (max_length - len(padded))
        rows.append(padded)
    return np.asarray(rows, dtype=np.int64)


def prepare_split_sequences(records, vocabulary: Vocabulary, config: CaptioningConfig) -> np.ndarray:
    sequences = [encode_caption(record.raw_caption, vocabulary, config) for record in records]
    return pad_encoded_sequences(sequences, vocabulary, config.text.max_caption_length)


def save_vocabulary_artifacts(vocabulary: Vocabulary, config: CaptioningConfig) -> None:
    output_dir = config.output.artifacts_dir / 'preprocessing'
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(vocabulary.token_to_id, output_dir / 'token_to_id.json')
    write_json({str(k): v for k, v in vocabulary.id_to_token.items()}, output_dir / 'id_to_token.json')
    write_json({'vocab_size': vocabulary.size, 'pad_id': vocabulary.pad_id, 'start_id': vocabulary.start_id, 'end_id': vocabulary.end_id, 'oov_id': vocabulary.oov_id, 'max_caption_length': config.text.max_caption_length}, output_dir / 'caption_metadata.json')
