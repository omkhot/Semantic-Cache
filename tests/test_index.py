import numpy as np
import pytest

from semcache.index import VectorIndex


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("backend", ["numpy", "auto"])
def test_add_search_remove(backend):
    idx = VectorIndex(3, backend)
    idx.add([1, 2, 3], np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([1, 1, 0])]))
    assert len(idx) == 3
    top = idx.search(_unit([1, 0.1, 0]), 2)
    assert top[0][0] == 1 and top[0][1] > 0.9
    idx.remove([1])
    assert len(idx) == 2
    assert 1 not in [i for i, _ in idx.search(_unit([1, 0, 0]), 5)]


def test_empty_search():
    assert VectorIndex(4, "numpy").search(np.ones(4, dtype=np.float32), 3) == []
