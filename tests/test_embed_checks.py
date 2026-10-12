"""The per batch vector check, which needs no GPU: silent failures become loud ones."""

import numpy as np
import pytest

from needtoknow.embed.model import EmbedError, check_batch


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
