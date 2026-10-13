# Online shortest-path teacher correction: CPU feasibility

Status: source/tokenizer feasibility, multimodal smoke, 64-update training, and fixed development256 evaluation are complete. The frozen gate failed and the method was stopped without scale-up.

## Existing simulator and action interface

The current VLN-CE `ShortestPathSensor` constructs a `ShortestPathFollower` with `return_one_hot=False` and returns one primitive action: STOP=0, MOVE_FORWARD=1, TURN_LEFT=2, TURN_RIGHT=3. The existing R2R text DSL parser accepts `stop`, `move forward` at 25/50/75 cm, and left/right turns at 15/30/45 degrees. Canonical one-step teacher strings can therefore be mapped without ambiguity to `stop`, `move forward 25cm`, `turn left 15 degrees`, and `turn right 15 degrees`.

The default sensor stop radius is 0.5 m. `OracleSuccess` uses geodesic distance `<3.0m`; a proposed teacher radius of 2.99m avoids an equality-boundary mismatch. The teacher label is privileged simulator supervision, not an inference-time semantic reward or a deployable policy component.

CPU-only inspection also loaded the actual Qwen tokenizer and checked the four canonical action strings. Their target token sequences are nonempty; the current chat template can mask target action tokens while excluding prompt and EOS tokens. Parser mapping and synthetic mask checks pass. This is not a multimodal trainer integration, real trajectory-label coverage audit, or gradient test.

The existing ActiveVLN recollection dataset stores observation, prior action, and oracle action records, but no registered ActiveVLN VLM DAgger trainer was found. The current VLM path carries per-turn multimodal context; adding a per-turn teacher CE loss requires new isolated integration. Existing action-match RLOO and sparse expert-prefix BC are distinct mechanisms and are not evidence for this dense supervised correction.

## Reuse/cost hypothesis

At each sampled policy trajectory, query the existing shortest-path follower at one uniformly sampled visited decision state and retain that exact multimodal context plus its canonical action string. This needs no external model/API, human labels, or additional image rendering by design. Actual incremental simulator latency, memory, valid-label coverage, and integration cost remain unmeasured and must be recorded in the smoke before scaling.

## Source identity inspected

SHA-256 values from the existing remote checkout:

- `VLN_CE/habitat_extensions/sensors.py`: `b2e46f024f5bdefe4f7d07d66e50333c94163151dd735cdeb7314c13130fb130`
- `VLN_CE/habitat_extensions/config/default.py`: `730072bf565fe0c288d145752e6419f484327cfc2594780f1ed62b55f0bf647d`
- `VLN_CE/habitat_extensions/measures.py`: `65dd459f7e17d86ce54271a2832e7efa33d810662b227570819ba795dfdcefd4`
- `utils/parse_utils.py`: `34797cafa2f230b178e9f71c64d691ccfcc05962ce333261322cd156e9f54bb2`
- `constants.py`: `f81e70ddcc9c572bd32428a1af9bc9468d4e505354de8ed56ae5b1e73eb8c2d6`
- `vlnce_baselines/common/recollection_dataset.py`: `61042aade2cd06ab70d6b8f1b4fd0d19613fc3e0e70d2b6282632650b2ee3dc6`



## Pilot evidence

The two-update multimodal smoke passed for candidate and control: 32 teacher targets/update, all four actions represented, nonempty action-only masks, finite actor gradients/KL, and nonzero candidate correction loss/gradient. Teacher-query latency was not separately instrumented; no runtime-efficiency claim is made.

Both 64-update arms then recorded 2,048 valid teacher targets (32/update), nonempty masks, finite KL, and nonzero actor gradients at all updates. On the fixed 256-episode/eight-scene development split, matched FP16 SFT achieved 109 successes (SR 42.58%, SPL 41.26%), RLOO control 52 (20.31%, 20.31%), and DAgger candidate 0 (0%, 0%). All arms had exact unique-ID coverage and zero inference errors. Candidate-minus-SFT paired SR/SPL were -42.58/-41.26 percentage points; candidate-minus-control were -20.31/-20.31 points. Independent raw-stat and local compact recounts agree. The frozen +2/+2 pp gate failed; do not open reserved/val-unseen or scale this method.

This is a single configured-seed result on an adaptively reused train-scene development split, not a clean generalization test. The teacher is privileged simulator supervision, not a deployed semantic signal.

The training wrapper checked for global_step_2 after a 64-step run; the evaluation wrapper wrote valid raw results outside the aggregator's expected subdirectory. Both original failure markers are preserved, unchanged source hashes and existing outputs were audited, and no inference was repeated during recovery.
