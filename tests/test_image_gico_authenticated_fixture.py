from __future__ import annotations

import numpy as np

from genode.artifacts.identity import semantic_sha256
from genode.gico.image_causal_stick import MAXIMUM_CLOCK_NODE_DRIFT, ImageGICOCausalPathBank

_FIXTURE_SHA256 = (
    "image-gico-causal-authenticated-test-fixture-v1:b581c9c8b20d58fd6918c7ccc21cccc1cf3b5148086d1535ac4989c0642c35c3"
)
_RAMP_TOKEN_PATH = np.asarray(
    [
        20,
        26,
        29,
        32,
        35,
        37,
        39,
        40,
        42,
        44,
        45,
        47,
        48,
        49,
        50,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        69,
        70,
        71,
        72,
        73,
        75,
        76,
        77,
        79,
        80,
        82,
        83,
        85,
        86,
        88,
        90,
        92,
        94,
        97,
        99,
        102,
        105,
        108,
        111,
        115,
        120,
        125,
        131,
        138,
        147,
        159,
        176,
        202,
    ],
    dtype=np.int64,
)


def _authenticated_support_fixture() -> tuple[np.ndarray, str]:
    uniform = np.ones(64, dtype=np.float64)
    ramp = np.arange(1, 65, dtype=np.float64)
    reverse = ramp[::-1].copy()
    center = 65.0 - np.abs(2.0 * np.arange(64) - 63.0)
    integer_families = (
        tuple(int(value) for value in uniform),
        tuple(int(value) for value in ramp),
        tuple(int(value) for value in reverse),
        tuple(int(value) for value in center),
    )
    support_rows = []
    for scale in (1.0, 2.0, 3.0):
        row = np.stack((uniform * scale, ramp * scale, ramp * scale, reverse * scale, center * scale))
        row /= row.sum(axis=-1, keepdims=True)
        support_rows.append(row)
    identity = semantic_sha256(
        {
            "protocol": "image-gico-causal-authenticated-test-fixture-v1",
            "integer_weight_families": integer_families,
        },
        namespace="image-gico-causal-authenticated-test-fixture-v1",
    )
    return np.stack(support_rows), identity


def test_authenticated_causal_fixture_has_stable_token_paths() -> None:
    support, fixture_sha256 = _authenticated_support_fixture()
    assert fixture_sha256 == _FIXTURE_SHA256

    bank = ImageGICOCausalPathBank.build(
        support,
        np.linspace(0.0, 1.0, 65, dtype=np.float64),
    )

    expected_ramp_paths = np.broadcast_to(_RAMP_TOKEN_PATH, bank.token_paths[:, 1].shape)
    assert np.array_equal(bank.token_paths[:, 1], expected_ramp_paths)
    assert np.array_equal(bank.token_paths[:, 1], bank.token_paths[:, 2])
    assert np.array_equal(bank.token_paths[0], bank.token_paths[1])
    assert np.array_equal(bank.token_paths[1], bank.token_paths[2])
    np.testing.assert_allclose(
        np.sum(bank.decoded_density_paths, axis=-1, dtype=np.float64),
        1.0,
        rtol=0.0,
        atol=8.0 * np.finfo(np.float64).eps,
    )
    assert bank.diagnostics.maximum_observed_clock_node_drift < MAXIMUM_CLOCK_NODE_DRIFT
    assert all(trie.children_by_prefix for trie in bank.tries)
