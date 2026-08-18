from __future__ import annotations

import numpy as np
import pytest

from genode.gico.image_conditional import _sha256_identity, validate_image_gico_density_alias_bindings


def test_sha256_lineage_requires_raw_or_lowercase_namespaced_digest() -> None:
    digest = "a" * 64

    assert _sha256_identity(digest, field="identity") == digest
    assert _sha256_identity(f"density-mass:{digest}", field="identity") == f"density-mass:{digest}"
    for invalid in ("density-0", "Density:" + digest, "density:" + "A" * 64):
        with pytest.raises(ValueError, match="lowercase SHA-256 identity"):
            _sha256_identity(invalid, field="identity")


def test_density_alias_identities_must_match_support_content_partition() -> None:
    support = np.asarray(
        [
            [[0.25, 0.75], [0.25, 0.75], [0.75, 0.25]],
            [[0.5, 0.5], [0.5, 0.5], [0.4, 0.6]],
            [[0.2, 0.8], [0.2, 0.8], [0.6, 0.4]],
        ],
        dtype=np.float64,
    )
    valid = tuple(("a" * 64, "a" * 64, "b" * 64) for _ in range(3))
    validate_image_gico_density_alias_bindings(valid, support)

    split_duplicate = [list(row) for row in valid]
    split_duplicate[0][1] = "c" * 64
    with pytest.raises(ValueError, match="aliases present in fixed_density_mass"):
        validate_image_gico_density_alias_bindings(split_duplicate, support)

    merge_distinct = [list(row) for row in valid]
    merge_distinct[1][2] = "a" * 64
    with pytest.raises(ValueError, match="aliases present in fixed_density_mass"):
        validate_image_gico_density_alias_bindings(merge_distinct, support)
