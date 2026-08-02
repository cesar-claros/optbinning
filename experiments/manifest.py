"""Artifact manifest generator (plan Sec. 9; re-review 4.9).

Scans the outputs tree and writes a machine-readable manifest so the
papers' reproducibility statements derive from an auditable table
rather than universal prose claims. Per artifact:

  - path, rows, columns (schema), file size, modified time;
  - resolved-configuration sidecar present (yes/no);
  - split/partition hash columns present (yes/no);
  - status-column summary where available.

Usage:
    python experiments/manifest.py [outputs_dir] [manifest_out]
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import json
import sys

from pathlib import Path

import pandas as pd


def build_manifest(outputs: Path) -> pd.DataFrame:
    rows = []
    for pq in sorted(outputs.rglob("*.parquet")):
        try:
            df = pd.read_parquet(pq)
        except Exception as err:                # pragma: no cover
            rows.append(dict(path=str(pq.relative_to(outputs)),
                             error=type(err).__name__))
            continue
        status = ""
        if "status" in df.columns:
            status = json.dumps(
                df.status.value_counts().to_dict(), sort_keys=True)
        rows.append(dict(
            path=str(pq.relative_to(outputs)),
            campaign=pq.parent.name,
            rows=len(df),
            n_columns=len(df.columns),
            columns=json.dumps(sorted(df.columns.tolist())),
            size_kb=round(pq.stat().st_size / 1024, 1),
            mtime=pd.Timestamp(pq.stat().st_mtime, unit="s")
            .isoformat(timespec="seconds"),
            config_sidecar=pq.with_suffix(".config.yaml").exists(),
            has_splits_hash="splits_hash" in df.columns,
            has_coordinate="coordinate" in df.columns,
            status_counts=status))
    return pd.DataFrame(rows)


def main() -> None:
    outputs = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs")
    out = Path(sys.argv[2] if len(sys.argv) > 2
               else outputs / "MANIFEST.parquet")
    man = build_manifest(outputs)
    man.to_parquet(out, index=False)
    n_side = int(man.get("config_sidecar", pd.Series(dtype=bool)).sum())
    print("manifest: {} artifacts, {} with config sidecars -> {}".format(
        len(man), n_side, out))
    man.drop(columns=["columns"]).to_csv(
        out.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
