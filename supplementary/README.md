# Supplementary evidence: version 2026-11-20

These files accompany **A Taxonomy of Architectural Transitions in Medical Image Segmentation: A Framework for Bottleneck Redistribution**.

## Contents

| File | Scope |
| --- | --- |
| method_register.csv | S1: 95 counted method families and 40 contextual/excluded records; 135 rows. Retrospective membership is unchanged; numerical-verification notes are synchronized with the revised S2. |
| numerical_verification.csv | S2: 68 source-specific numerical records, revised on 20 November 2026 while preserving stable IDs and prior history. |
| nv_cell_audit_2026-11-20.csv | All 72 previously withheld numerical entries: 26 numerical recoveries, 33 inspected-source nonreports, 12 unresolved values, and one unsupported dataset entry removed. |
| contextual_update_2026.csv and .bib | 23 separately identified CVPR/MICCAI 2026 contextual publications; excluded from the frozen S1 count. |
| Supplementary_Information.pdf and .tex | Interpretation, status definitions, unresolved questions, and scope. |
| SHA256SUMS.txt | Exact file hashes for this snapshot. |
| CITATION.cff | Citation metadata for the supplementary compilation. |

## Interpretation

The retrospective literature boundary is 30 June 2026. The November update is targeted, not an exhaustive new search or a reconstructed PRISMA screening log. Numerical provenance and comparability are separate. Later evaluation values are explicitly attributed in the manuscript and S2.

**NR** means no exact-configuration value was reported in the inspected source version. It is not a claim that no later paper or implementation reports one. **NV** means verification remains unresolved because of source access, variant attribution, or units. Neither means zero. Classifying a missing measurement as NR does not recover a number.

No patient data, trained models, third-party full papers, new experiments, or independent model profiling are included. Computational costs retain the source's accounting limits. A transcription of H2Former's original article was inspected through a mirror; that access limitation is documented.

## Unresolved measurements

Twelve numerical entries remain NV: CBAM-U-Net (four), SegMamba-V2 (four), FCT-384 costs (two), and FAT-Net/BATFormer operation-count units (one each). MIST's retained source aggregate also differs from the simple mean of its printed organ values; this separate aggregation issue remains documented.

## Version and reuse

Version: 2026-11-20. The accompanying MIT license is copied from the existing project repository. This compilation does not change rights in the cited publications. Cite the manuscript for its interpretation and the exact version of these records when using the audit. A GitHub version is not a DOI; no DOI is claimed by this package.
