"""One-time, read-only archive extraction. Simulator never calls legacy code."""
from pathlib import Path
import hashlib
import json
import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / "results/pool_cache.npz"
    metadata_source = root / "results/pool_cache.json"
    metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    indices = [34, 6, 26]
    out = root / "journal_sim/tests/fixtures"
    out.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        np.savez_compressed(out / "conference_representatives.npz", indices=np.array(indices),
                            **{key: data[key][indices] for key in ("w", "theta", "H", "sinr")})
    payload = dict(source_npz_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   source_json_sha256=hashlib.sha256(metadata_source.read_bytes()).hexdigest(),
                   source_path="results/pool_cache", indices=indices,
                   candidates=[metadata["candidates"][i] for i in indices], meta=metadata["meta"],
                   model_note="Archived conference channels: direct intercell OFF. No new generation or recertification by legacy code.")
    (out / "conference_representatives.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Extracted archived candidates", indices)


if __name__ == "__main__":
    main()
