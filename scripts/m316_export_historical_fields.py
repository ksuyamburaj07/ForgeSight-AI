"""Create browser-readable historical patch evidence.

Source: retained four-case M3.13C diagnostic array archive.
This is a derived visualization export, not a new evaluation.

The archive's score agreement does not independently
establish its execution-environment provenance.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ARCHIVE = ROOT / (
    "reports/diagnostics/m314b3/"
    "m313c_all_four_patch_arrays_v1.npz"
)

DEST = ROOT / "frontend/public/research-atlas"

EXPECTED_SHA = (
    "cbb58d6aafbfd9e62570f67a9050f449a64"
    "de106f58872a3d3459c426aa21e3f"
)

EXPECTED_COUNTS = {
    "B3-01": 1728,
    "B3-09": 1728,
    "B3-17": 1440,
    "B3-23": 1872,
}

if not ARCHIVE.is_file():
    raise SystemExit("STOP: Historical array archive missing")

actual_sha = hashlib.sha256(
    ARCHIVE.read_bytes()
).hexdigest()

if actual_sha != EXPECTED_SHA:
    raise SystemExit(
        "STOP: Frozen archive fingerprint mismatch"
    )

manifest_path = DEST / "cases.json"

if not manifest_path.is_file():
    raise SystemExit("STOP: Case Atlas manifest missing")

manifest = json.loads(
    manifest_path.read_text(encoding="utf-8")
)

records = {
    item["id"]: item
    for item in manifest["cases"]
}

if set(records) != set(EXPECTED_COUNTS):
    raise SystemExit("STOP: Case manifest identities mismatch")

outputs = {}

with np.load(ARCHIVE, allow_pickle=False) as data:
    for case_id, expected_count in EXPECTED_COUNTS.items():
        prefix = case_id.replace("-", "_")

        distances = data[
            f"{prefix}__distance_map"
        ]
        mask = data[
            f"{prefix}__valid_mask"
        ]

        if (
            distances.shape != (48, 48)
            or mask.shape != (48, 48)
        ):
            raise SystemExit(
                f"STOP: Invalid grid shape for {case_id}"
            )

        if not np.isin(mask, (0, 1)).all():
            raise SystemExit(
                f"STOP: Invalid mask for {case_id}"
            )

        valid = distances[mask == 1].astype(
            np.float64
        )

        if (
            len(valid) != expected_count
            or not np.isfinite(valid).all()
            or np.any(valid < 0)
        ):
            raise SystemExit(
                f"STOP: Invalid distances for {case_id}"
            )

        k = math.ceil(len(valid) * 0.01)

        calculated = float(
            np.partition(valid, -k)[-k:].mean()
        )

        reference = float(
            records[case_id]["reproduction"][
                "frozen_score"
            ]
        )

        # Consistency bound for this derived export.
        # NOT the frozen CPU/GPU acceptance criterion.
        if abs(calculated - reference) > 1e-6:
            raise SystemExit(
                f"STOP: Score consistency check failed: "
                f"{case_id}"
            )

        distance_grid = [
            [
                (
                    float(distances[r, c])
                    if mask[r, c] == 1
                    else None
                )
                for c in range(48)
            ]
            for r in range(48)
        ]

        payload = {
            "schema_version": 1,
            "case_id": case_id,
            "grid": [48, 48],
            "archive_sha256": actual_sha,
            "evidence_source": (
                "retained_m313c_diagnostic_array_archive"
            ),
            "execution_origin_independently_verified": False,
            "valid_count": int(len(valid)),
            "top_k_count": int(k),
            "archived_top_k_mean": calculated,
            "frozen_reference_score": reference,
            "distances": distance_grid,
            "valid_mask": (
                mask.astype(bool).tolist()
            ),
            "interpretation": (
                "Historical cosine nearest-neighbor "
                "patch-distance evidence. Numeric agreement "
                "with frozen reference scores does not "
                "independently prove execution provenance."
            ),
        }

        filename = f"{case_id}_patch_field.json"

        path = DEST / filename

        if path.exists():
            raise SystemExit(
                f"STOP: Refusing to overwrite {filename}"
            )

        outputs[path] = (
            json.dumps(
                payload,
                separators=(",", ":"),
                allow_nan=False,
            ) + "\n"
        )

# All scientific checks and output-path checks pass
# before the first file is written.
for path, content in outputs.items():
    with path.open("x", encoding="utf-8") as file:
        file.write(content)

print("=== HISTORICAL PATCH EXPORT COMPLETE ===")
print("Pinned archive SHA-256:", actual_sha)

for path in outputs:
    print(path.relative_to(ROOT))

print("No original research arrays modified.")
print("Archive execution provenance remains qualified.")
