"""The per batch vector check, which needs no GPU: silent failures become loud ones."""

import numpy as np
import pytest

from needtoknow.embed.model import EmbedError, batches, check_batch


def unit_rows(count: int) -> np.ndarray:
    rows = np.random.default_rng(0).normal(size=(count, 1024)).astype(np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_a_sound_batch_passes_through_unchanged():
    rows = unit_rows(4)
    assert check_batch(rows) is rows


def test_a_batch_with_a_nan_is_refused():
    rows = unit_rows(4)
    rows[2, 7] = np.nan
    with pytest.raises(EmbedError, match="NaN or infinite"):
        check_batch(rows)


@pytest.mark.parametrize("scale", [0.0, 0.5, 2.0])
def test_a_row_that_is_not_unit_length_is_refused(scale):
    rows = unit_rows(4)
    rows[1] *= scale
    with pytest.raises(EmbedError, match="norms from"):
        check_batch(rows)


def test_batches_hold_every_index_once_and_never_pad_past_the_budget():
    lengths = [int(n) for n in np.random.default_rng(1).integers(3, 490, size=2000)] + [5000]
    cut = batches(lengths, 4096)
    assert sorted(i for batch in cut for i in batch) == list(range(len(lengths)))
    assert all(len(batch) * max(lengths[i] for i in batch) <= 4096 for batch in cut if len(batch) > 1)
    assert [2000] in cut
    assert all(lengths[b[i]] <= lengths[b[i + 1]] for b in cut for i in range(len(b) - 1))
