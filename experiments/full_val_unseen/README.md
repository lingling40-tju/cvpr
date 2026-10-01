# Complete R2R val-unseen evaluation

These are measured results from `wanghaozhihuoshanyun` at
`/Knowin/foundation/haozhiwang/whz/ActiveVLN_semantic_20260930/runlogs/eventtrace_full_val_unseen`.
The evaluation finished on 2026-10-01 (UTC). Seven models each ran on exactly the
1,839 episode IDs in `manifest.json`, spanning 11 scenes. The source dataset,
checkpoint preparation, and evaluation commands are documented by the scripts
in the parent `experiments/` directory. Four Habitat shards evaluated each
model. `*.validated.json` and `*.completed` are the per-model completion gates;
all report zero inference errors.

`episodes.csv` is a compact export of all 12,873 per-episode simulator metric
records, in manifest order within each model. It contains the model label,
episode and scene IDs, success, SPL, terminal goal distance, path length,
oracle success, and early-stop reason. Run `python3 verify_export.py` here to
recompute summaries and paired contrasts from the CSV. `analysis.json` contains arm summaries
and paired seed differences; `evaluator_logs.tar.gz` retains the concise
`eval_*_shard*.log` and `*.run.log` console logs. Large vLLM request logs and
raw image data are not included. The export script is
`../export_full_val_compact.py`.

Success rate differences for EventTrace minus destination-only control are
+0.65, +0.44, and -2.88 percentage points at seeds 11, 22, and 33. The mean
paired difference is -0.60 points, with a 1.98-point sample standard deviation
over the three seeds. This does not support a consistent navigation gain.

The val-unseen evaluator generates one trajectory per model and episode with
temperature 0.2, top-p 0.8, a 12-turn limit, and a 76,800-pixel image cap.
Episode IDs and settings are matched, but decoding is stochastic. These
measurements do not test verifier accuracy; the separate blind review has
only one AI labeler and no independent human adjudication.
