"""Disk-backed forward-field chunking for adjoint overlap gradients."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MemmapConfig:
    """Configuration for memmap chunk storage."""

    directory: str
    chunk_size: int = 64
    dtype: str = "float32"


class ForwardFieldChunkWriter:
    """Write forward fields to deterministic memmap chunks on disk."""

    def __init__(
        self,
        config: MemmapConfig,
        field_names: tuple[str, ...],
        field_shape: tuple[int, ...],
    ):
        self.config = config
        self.field_names = field_names
        self.field_shape = tuple(field_shape)
        self.dtype = np.dtype(config.dtype)
        self.path = Path(config.directory)
        self.path.mkdir(parents=True, exist_ok=True)

        self._buffers = {name: [] for name in field_names}
        self._chunk_lengths: list[int] = []
        self._chunk_count = 0
        self._total_steps = 0

    def append(self, fields: dict[str, np.ndarray]):
        """Append one forward step into the chunk buffers."""
        for name in self.field_names:
            arr = np.asarray(fields[name], dtype=self.dtype)
            if arr.shape != self.field_shape:
                raise ValueError(
                    f"Field {name!r} shape {arr.shape} != expected {self.field_shape}"
                )
            self._buffers[name].append(arr)

        self._total_steps += 1
        if len(self._buffers[self.field_names[0]]) >= self.config.chunk_size:
            self.flush()

    def flush(self):
        """Flush current buffers to disk as one chunk."""
        if not self._buffers[self.field_names[0]]:
            return

        chunk_len = len(self._buffers[self.field_names[0]])
        chunk_id = self._chunk_count

        for name in self.field_names:
            out_path = self.path / f"chunk_{chunk_id:06d}_{name}.dat"
            mm = np.memmap(
                out_path,
                mode="w+",
                dtype=self.dtype,
                shape=(chunk_len, *self.field_shape),
            )
            mm[:] = np.stack(self._buffers[name], axis=0)
            mm.flush()

        self._chunk_lengths.append(chunk_len)
        self._chunk_count += 1
        for name in self.field_names:
            self._buffers[name].clear()

    def finalize(self):
        """Flush remaining data and write metadata."""
        self.flush()
        metadata = {
            "field_names": list(self.field_names),
            "field_shape": list(self.field_shape),
            "dtype": self.dtype.str,
            "chunk_lengths": self._chunk_lengths,
            "total_steps": self._total_steps,
        }
        (self.path / "metadata.json").write_text(json.dumps(metadata, indent=2))


class ReverseFieldChunkReader:
    """Read forward field chunks in reverse order for adjoint accumulation."""

    def __init__(self, directory: str, delete_consumed: bool = False):
        self.path = Path(directory)
        meta_path = self.path / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing memmap metadata: {meta_path}")

        self.meta = json.loads(meta_path.read_text())
        self.field_names = tuple(self.meta["field_names"])
        self.field_shape = tuple(self.meta["field_shape"])
        self.dtype = np.dtype(self.meta["dtype"])
        self.chunk_lengths = list(self.meta["chunk_lengths"])
        self.total_steps = int(self.meta["total_steps"])
        self.delete_consumed = delete_consumed

    def iter_chunks_reverse(self):
        """Yield chunk dicts {field_name: memmap} from latest to earliest."""
        for chunk_idx in range(len(self.chunk_lengths) - 1, -1, -1):
            length = int(self.chunk_lengths[chunk_idx])
            chunk = {}
            paths = []
            for name in self.field_names:
                path = self.path / f"chunk_{chunk_idx:06d}_{name}.dat"
                paths.append(path)
                chunk[name] = np.memmap(
                    path,
                    mode="r",
                    dtype=self.dtype,
                    shape=(length, *self.field_shape),
                )
            yield chunk_idx, chunk

            if self.delete_consumed:
                for path in paths:
                    path.unlink(missing_ok=True)


def compute_overlap_gradient_memmap(
    reader: ReverseFieldChunkReader,
    adjoint_history: Iterable[np.ndarray],
    field_key: str = "Ez",
) -> np.ndarray:
    """Accumulate standard overlap gradient from memmap forward history.

    Gradient = sum_t forward[t] * adjoint[T-1-t]
    """
    adj = [np.asarray(a) for a in adjoint_history]
    if len(adj) == 0:
        raise ValueError("adjoint_history must not be empty")

    grad = np.zeros(reader.field_shape, dtype=np.float64)
    t_global = reader.total_steps - 1

    for _chunk_id, chunk in reader.iter_chunks_reverse():
        fwd = np.asarray(chunk[field_key])
        for i in range(fwd.shape[0] - 1, -1, -1):
            if t_global < 0:
                break
            adj_idx = reader.total_steps - 1 - t_global
            if adj_idx >= len(adj):
                break
            grad += fwd[i] * adj[adj_idx]
            t_global -= 1

    return grad.astype(np.float32)
