# Archive

Superseded plans, closed investigations, executed design records and dated run notes. It is kept
because several of this project's conclusions were reversed on evidence, and a reader evaluating
the final claims is better served by seeing the reversals than by a tidied-up record.

**Nothing here describes the current state of the code, with one qualification.** The documents
under *Executed plans and specifications* are the design records for work that shipped, and some
are cited by the shipped code section-by-section — `rescore.py` names its specification,
`mulan/interface_xattn.py` names its plan. Those documents remain accurate about **why** the code
is shaped as it is; they are archived because they are written as forward plans, and their status
lines, queue state and "next steps" describe a moment that has passed. Where such a document
disagrees with the code, the code is right.

Every other section here is superseded outright. **Membership of this directory is the status**:
a document being here means it does not describe the current state, whether or not it says so
inside. Many carry a status banner naming what replaced them, and those are the reliable ones to
read first — but a document without a banner is not thereby current, and several of the oldest
have none. Where a claim here matters, check it against the document named in the section heading
below rather than against the archived text.

Archived documents may reference paths in the local `scratch/` working tree, which is not
published. Those references are historical and are not expected to resolve.

## Superseded by a current document

| Archived | Superseded by | Why |
|---|---|---|
| [`FOLDX_SUMMARY.md`](FOLDX_SUMMARY.md) | [`../FOLDX_STRUCTURE_ARC_AND_AUGMENTATION_PLAN.md`](../FOLDX_STRUCTURE_ARC_AND_AUGMENTATION_PLAN.md) | Written before the FoldX-alone baseline existed. Its central "+0.014–0.019 PCC" framing is exactly what the later document revises. Retained for the chronology of how FoldX entered the project, and for its terminology appendices. |
| [`PLM_COMPARISON_S1102.md`](PLM_COMPARISON_S1102.md) | [`../RESULTS.md`](../RESULTS.md) | Early Ankh-vs-ProstT5 findings, folded into the consolidated results. |
| [`PROSTT5_STRUCTURE_OPTIONS.md`](PROSTT5_STRUCTURE_OPTIONS.md), [`PLAN_PROSTT5_STRUCTURE_v2.md`](PLAN_PROSTT5_STRUCTURE_v2.md) | [`../FOLDX_STRUCTURE_ARC_AND_AUGMENTATION_PLAN.md`](../FOLDX_STRUCTURE_ARC_AND_AUGMENTATION_PLAN.md) | Route-by-route structure plans, subsumed by the arc document that evaluates all of them. |

## Closed investigations

| Archived | Outcome |
|---|---|
| [`FINDINGS_FOLDX_MUTANT_3DI.md`](FINDINGS_FOLDX_MUTANT_3DI.md) | **Negative result.** FoldX-built mutant structures do not yield mutation-aware 3Di tokens. This is what redirected FoldX from a structure-generator to a ΔΔG score channel — the pivot behind the whole `foldx` arm. |
| [`FOLDENV_EXTRACTION_PLAN.md`](FOLDENV_EXTRACTION_PLAN.md) | **Executed in full.** The structural-context module was extracted to [foldenv](https://github.com/cchin29/foldenv), and its final phase — removing the module from this repository and rewiring the consumers — is now done too. |
| [`CODE_REVIEW_0712.md`](CODE_REVIEW_0712.md) | Point-in-time code review; most items addressed. |
| [`PLAN_MERGE_MPS_ANKH3.md`](PLAN_MERGE_MPS_ANKH3.md) | One-off merge plan, executed. |

## Plans not carried out as written

| Archived | Status |
|---|---|
| 2026-07-09 broad plan | *Withdrawn before publication, not archived here.* A remaining-work plan that self-declared stale in its opening lines and referenced scaffolding that is not published with this repository. Documents elsewhere still cite its section numbers as provenance; the sections are not recoverable from this tree. |
| [`PLAN_AIDO_CPU_EXECUTION.md`](PLAN_AIDO_CPU_EXECUTION.md) | Detailed AIDO-16B CPU/GPU execution runbook, written for a second machine. The runs completed; the runbook is machine-specific. |

## Executed plans and specifications

Design records for work that ran to completion. The results they produced are in
[`../RESULTS.md`](../RESULTS.md) and the `SUMMARY_*.md` files beside the code.

| Archived | What it planned | Where the result is |
|---|---|---|
| [`PLAN_FULL_SKEMPI.md`](PLAN_FULL_SKEMPI.md) | The move from S1102 to full SKEMPI v2 — row curation, three split protocols, both FoldX arms | `data/splits/splits_skempi_full_*`, `experiments/full_skempi_seqonly/SUMMARY_*.md`. Builders and scorers there cite it by section. |
| [`PLAN_RETRAIN_BYCOMPLEX.md`](PLAN_RETRAIN_BYCOMPLEX.md) | Leakage-controlled retrain on by-complex and homology-clustered folds | `experiments/retrain_split/SUMMARY.md`, `SUMMARY_clustered.md`. Its drivers cite it by section. |
| [`RESCORE_PERSTRUCTURE_SPEC.md`](RESCORE_PERSTRUCTURE_SPEC.md) | Per-structure Pearson/Spearman + AUROC re-scoring of existing OOF predictions | `experiments/rescore_perstructure/rescore.py`, which names this file as what it implements. |
| [`PLAN_A1_INTERFACE_XATTN.md`](PLAN_A1_INTERFACE_XATTN.md) | Residue-level interface cross-attention in the head | `mulan/interface_xattn.py`, wired through `mulan/modules.py`. |
| [`PLAN_MINT_A2.md`](PLAN_MINT_A2.md) | MINT partner-context embeddings, to attack the cross-chain electrostatic blind spot | Ran; MINT does not beat the multi-PLM consensus and is weakest on the tail it targeted. `mulan/data.py` cites §2 for the pair-cache contract. |
| [`PLAN_AIDO.md`](PLAN_AIDO.md) | AIDO.Protein-16B as the sequence embedder | [`../RESULTS.md`](../RESULTS.md) §14 — the largest encoder ties Ankh. Registered in `mulan/plm/registry.py`, not by the `constants.py` edit the plan proposes. |

## Assessments overturned by measurement

| Archived | Outcome |
|---|---|
| [`SAPROT_ASSESSMENT.md`](SAPROT_ASSESSMENT.md) | Recommended skipping SaProt on the reasoning that WT-only structure input resembled ProstT5's negative 3Di channel. SaProt was run anyway and leads several benchmarks — [`../RESULTS.md`](../RESULTS.md) §15, §17. Kept for the reasoning, which is where the error is legible. |

## Dated result snapshots

[`AIDO_LADDER_RESULTS_20260722.md`](AIDO_LADDER_RESULTS_20260722.md), [`AIDO_S1102_BALANCED_RESULTS.md`](AIDO_S1102_BALANCED_RESULTS.md) — point-in-time tables for the
AIDO-16B campaigns, kept because they record per-fold detail that the consolidated results
summarize away.

## `run_notes/`

Per-campaign result notes that lived alongside their run outputs in the working tree — the
earliest records of the S1102, ProstT5, layer-probe and augmentation runs.

| Note | Campaign |
|---|---|
| [`RESULTS.md`](run_notes/RESULTS.md) | The first consolidated run record |
| [`results_RESULTS.md`](run_notes/results_RESULTS.md) | run1 — S1102 / Ankh |
| [`results_run2_s1102_prostt5_RESULTS.md`](run_notes/results_run2_s1102_prostt5_RESULTS.md) | run2 — S1102 / ProstT5 |
| [`results_run3a_prostt5_L7_RESULTS.md`](run_notes/results_run3a_prostt5_L7_RESULTS.md) | run3a — ProstT5, layer 7 |
| [`results_run3b_prostt5_L7-10_RESULTS.md`](run_notes/results_run3b_prostt5_L7-10_RESULTS.md) | run3b — ProstT5, layers 7–10 |
| [`results_run4a_ankh_aug_RESULTS.md`](run_notes/results_run4a_ankh_aug_RESULTS.md) | run4a — Ankh + augmentation |
| [`results_run4b_prostt5_aug_RESULTS.md`](run_notes/results_run4b_prostt5_aug_RESULTS.md) | run4b — ProstT5 + augmentation |
| [`results_probe_layers_RESULTS.md`](run_notes/results_probe_layers_RESULTS.md) | Layer-probe raw tables |
| [`TRAINING_CONCEPTS.md`](run_notes/TRAINING_CONCEPTS.md) | Explainer of how MuLAN training works — the one file here a reader may mistake for current documentation |

## Commit hashes cited in this repository

Several current documents cite pre-squash commit SHAs as provenance — `docs/RESULTS.md`'s
"Bug fixed (commit `043ac4a`)", and similar in `experiments/`. **None of them resolve here.** This
repository is a squashed republication: its history begins at one commit, and the history those
hashes belong to is deliberately not published. They are kept because the sentence beside each one
says what changed, which is the part a reader needs; the hash is a pointer into an archive only the
maintainer holds. Treat an unresolvable SHA in this repository as a date-stamp, not a broken link.

## Machine handoff runbooks

`runbooks/` holds the operational records of how runs were produced — private machines, branches
that no longer exist, `scratch/` paths. See [`runbooks/README.md`](runbooks/README.md).
