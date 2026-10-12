# Completed three-method development comparison

This archive contains the fixed evaluation identity, exact development manifest, the five compact paired-episode exports, original analyzer reports and compact recounts, the final raw internal-ID verification report, and the three 64-step training audits for the group-four pilot described in `../../README.md`.

The fixed screen contains 256 R2R-train episodes from eight scenes. All four evaluated policies (the reused FP16 SFT plus three candidates) had exact coverage and zero inference errors. The remote final verifier checked 1,024 raw evaluator-internal IDs and recomputed all five paired comparisons; its PASS report and SHA-256 inventory are included. This local package intentionally omits the full Habitat shard outputs; the compact per-episode comparisons and the verifier’s raw-file SHA inventory are retained.

The standard-library `verify_compact_local.py` independently recounts all five compact pairs from the rows, checks their IDs against the frozen manifest, checks reports and raw-ID aggregate provenance, and validates the failed scale gates. Run:

```sh
python3 verify_compact_local.py
```

Results are exploratory: one configured training seed, an adaptively reused development screen, and no established rollout-stream independence. Turn-level RLOO and the SRGPO-style composite use privileged simulator geometry; the latter also changes outcome normalization, so its effect cannot be attributed to process grouping alone.
