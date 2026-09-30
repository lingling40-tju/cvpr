"""Build the same eight curated episodes for control and semantic reward arms."""

from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path("/Knowin/foundation/haozhiwang/whz/ActiveVLN_semantic_20260930")
SOURCE = ROOT / "data/r2r_4000_train.parquet"
OUTPUT = ROOT / "data/r2r_semantic_pilot8.parquet"
EPISODE_IDS = [2889, 2890, 7110, 8033, 943, 2298, 2412, 10440]


def main():
    table = pq.read_table(SOURCE)
    rows = table.to_pylist()
    by_id = {int(row["extra_info"]["episode_id"]): idx for idx, row in enumerate(rows)}
    assert all(episode_id in by_id for episode_id in EPISODE_IDS)
    pq.write_table(table.take([by_id[episode_id] for episode_id in EPISODE_IDS]), OUTPUT)
    print(OUTPUT)
    print(EPISODE_IDS)


if __name__ == "__main__":
    main()
