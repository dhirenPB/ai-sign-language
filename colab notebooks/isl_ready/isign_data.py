"""
PyTorch data pipeline for iSign shards (replaces the torchtext Field/Iterator code).
Written into the Progressive Transformers repository by the notebook; import as `isign_data`.
"""
import re
from collections import Counter
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

import isign_pose as ip
from constants import DEFAULT_UNK_ID, EOS_TOKEN, PAD_TOKEN
from vocabulary import Vocabulary

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text):
    """Lower-case and split into words and punctuation marks."""
    return _TOKEN_RE.findall(text.lower())


class ISignShardDataset(Dataset):
    """
    Sequences stored by isign_pose.convert_manifest. Each item holds
      src            list of tokens
      trg            float32 [T, 226]   (225 normalised coordinates + frame counter)
      landmark_valid bool    [T, 75]    (False where a landmark is missing / unreliable)
      file_paths     the uid
    """

    def __init__(self, shard_dir, normalization=None, max_shards=None, max_frames=None):
        expected = ip.params_hash(normalization) if normalization is not None else None
        self.store = ip.ShardStore(shard_dir, expected_hash=expected, max_shards=max_shards)
        if len(self.store) == 0:
            raise ValueError(f"No converted shards found in {shard_dir}")
        if max_frames is not None and int(self.store.lengths.max()) > max_frames:
            raise ValueError(f"A sequence has {int(self.store.lengths.max())} frames (> max_frames={max_frames})")
        self._tokens = None

    def __len__(self):
        return len(self.store)

    def token_lists(self):
        if self._tokens is None:
            self._tokens = [tokenize(t) for t in self.store.texts]
        return self._tokens

    def __getitem__(self, idx):
        uid, text, coords, valid = self.store.get(int(idx))
        tokens = self.token_lists()[int(idx)] if self._tokens is not None else tokenize(text)
        return {
            "src": tokens,
            "trg": torch.from_numpy(ip.add_counter(coords)),
            "landmark_valid": torch.from_numpy(valid),
            "file_paths": uid,
        }


def build_vocab(token_lists, min_freq=1, max_size=None):
    """Source vocabulary (specials + words), most frequent first."""
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    words = [t for t, c in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])) if c >= min_freq]
    if max_size is not None:
        words = words[:max_size]
    return Vocabulary(tokens=words)


def collate_isign(examples, vocab):
    """Pad examples into the batch object that the repository's Batch class expects."""
    unk_id = DEFAULT_UNK_ID()
    pad_id = vocab.stoi[PAD_TOKEN]
    eos_id = vocab.stoi[EOS_TOKEN]

    src_seqs = [torch.tensor([vocab.stoi.get(t, unk_id) for t in ex["src"]] + [eos_id], dtype=torch.long)
                for ex in examples]
    src_lengths = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    src = pad_sequence(src_seqs, batch_first=True, padding_value=pad_id)

    frame_lengths = torch.tensor([ex["trg"].shape[0] for ex in examples], dtype=torch.long)
    trg = pad_sequence([ex["trg"] for ex in examples], batch_first=True, padding_value=0.0)
    # padded frames are 'not valid' for every landmark
    landmark_valid = pad_sequence([ex["landmark_valid"].to(torch.uint8) for ex in examples],
                                  batch_first=True, padding_value=0).bool()

    return SimpleNamespace(
        src=(src, src_lengths),
        trg=trg,
        frame_lengths=frame_lengths,
        landmark_valid=landmark_valid,
        file_paths=[ex["file_paths"] for ex in examples],
    )


class EpochShuffleSampler(Sampler):
    """Shuffles with seed + epoch, so a resumed run sees the same order as an uninterrupted one."""

    def __init__(self, n, seed):
        # no super().__init__(): Sampler's constructor does nothing, and its signature differs between torch versions
        self.n, self.seed, self.epoch = int(n), int(seed), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        return iter(np.random.RandomState(self.seed + self.epoch).permutation(self.n).tolist())

    def __len__(self):
        return self.n


def make_loader(dataset, vocab, batch_size=8, shuffle=False, seed=27, num_workers=0, pin_memory=False):
    sampler = EpochShuffleSampler(len(dataset), seed) if shuffle else None
    # the workers only need the token -> id table; a plain dict pickles under every start method (spawn / forkserver)
    lite_vocab = SimpleNamespace(stoi=dict(vocab.stoi))
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
                      collate_fn=partial(collate_isign, vocab=lite_vocab),
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=num_workers > 0)       # on Windows every new worker process costs seconds
