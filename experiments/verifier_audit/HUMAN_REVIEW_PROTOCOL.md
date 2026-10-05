# Blinded human review of the selected EventTrace transitions

Status (2026-10-05): two independent human reviewers have returned complete
CSV files. They agree on 33/49 three-way labels (Cohen's kappa 0.459), with
16 disagreements still unadjudicated. The answer-free reviewer package and
the aggregate, checksummed analysis are retained separately. This is a selected, transition-enriched pilot,
not a population accuracy study. The source contains 49 transitions from
12 R2R val-unseen episodes; the source bundle and prior single-AI review
remain separate from the reviewer package.

The locally prepared reviewer zip is
`/Users/wanghaozhi/Documents/ChatGPT/vln/eventtrace-blind-human-review-49.zip`
(SHA-256 `37b193bd0642b2e501ab81e499032ad96521d541abeadf3aa307fe320ffad924`).
It holds a randomized offline HTML form, 98 byte-verified JPEGs, and a
blank CSV. It contains **no** model verdict, previous AI label, original
episode ID, original image name, or answer key. The mapping key is kept
outside this repository and must not be given to reviewers. The package
uses only the already selected audit images; it introduces no new val-unseen
selection or experiment.

The same answer-free cases were also provided as a single offline HTML file
with embedded images and CSV export. The independent-review agreement report
is in [`../turn_rloo_20261005/human_review_agreement.json`](../turn_rloo_20261005/human_review_agreement.json).
The reviewers' free-text CSVs remain outside the public repository. Until a
third independent blind adjudication, these files support agreement and
rubric analysis only, not model accuracy or a confusion table against human
ground truth.

## Review procedure

1. Recruit **two independent human reviewers** and give each only a copy
   of the zip. They should not inspect this repository's audit bundle,
   model outputs, simulator geodesics, each other's labels, or any
   previous review.
2. Each reviewer reads the full instruction, event clause, actions and
   before/after images. Label **Y** only when the specified event is
   clearly completed, **N** only when clearly not completed, and **U**
   when the provided views and action trace cannot decide. For a stop
   event, the stop action alone does not establish proximity. For an
   event that needs earlier unseen route history, choose U unless the
   supplied evidence resolves it. Record confidence 1–3 and a brief
   evidence note.
3. Lock both independent CSVs before comparison. Report three-way raw
   agreement and Cohen's kappa. A third human reviewer adjudicates each
   disagreement while blind to the model prediction and prior AI label;
   retain both original labels and the adjudication.
4. Join the adjudicated labels to the model's 49 verdicts using the
   private key. Report the complete Y/N/U confusion table and the count
   of uncertain labels. Compute completed precision only among model-Y
   cases with adjudicated Y or N, and completed recall only among
   adjudicated-Y cases; show both denominators. Do not count an
   adjudicated U as a false positive or negative. Do not tune the
   verifier, prompt, or reward using these labels.

Even after review, results describe this deliberately selected set.
They cannot establish population verifier precision, semantic reward
benefit, or navigation improvement. A separately sampled and blinded
evaluation would be needed for those claims. The
[`build_blind_human_review.py`](build_blind_human_review.py) generator
checks source image hashes and keeps the package answer-free.
