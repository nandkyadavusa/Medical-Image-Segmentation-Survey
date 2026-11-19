# Additional medical image segmentation papers — 2026

**Curated reading-list update:** 19 November 2026  
**Scope:** 12 papers, verified against publisher/conference proceedings landing pages.  
**Status:** supplementary reading list only; it does **not** amend the manuscript's frozen 95-family method register or Supplementary Data S1/S2.

The companion manuscript's retrospective literature boundary is **30 June 2026**. Three CVPR 2026 papers below were available before that date but were not part of the original named-method inventory. Nine MICCAI 2026 papers were published after the boundary and must not silently be added to the original 95-method count. Inclusion in this page is **not** a claim of experimental replication, independent numeric verification, or eligibility for a matched cross-paper leaderboard.

**Objective shorthand:** LP = local spatial precision; GC = global contextual reasoning; CE = computational efficiency; CG = cross-domain generalization. These mappings indicate research relevance, **not validated measurements** of those latent constructs.

## Reference list

| ID | Paper | Venue | Main mechanism | Relevant objectives | Code |
| --- | --- | --- | --- | --- | --- |
| P26-01 | [PMRNet: Physics-informed Multi-scale Refinement Network for Medical Image Segmentation](https://openaccess.thecvf.com/content/CVPR2026/html/Kang_PMRNet_Physics-informed_Multi-scale_Refinement_Network_for_Medical_Image_Segmentation_CVPR_2026_paper.html) — Boce Kang | CVPR 2026 | Physics-informed multi-scale CNN; boundary-aware decoding and efficient context propagation | LP;GC;CE | [Link](https://github.com/KangBoce/PMRNet) |
| P26-02 | [MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation](https://openaccess.thecvf.com/content/CVPR2026/html/Koleilat_MedCLIPSeg_Probabilistic_Vision-Language_Adaptation_for_Data-Efficient_and_Generalizable_Medical_Image_CVPR_2026_paper.html) — Taha Koleilat et al. | CVPR 2026 | Probabilistic cross-modal image–text attention; uncertainty-aware vision-language adaptation | LP;CG | Not confirmed |
| P26-03 | [VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation](https://openaccess.thecvf.com/content/CVPR2026/html/Rokuss_VoxTell_Free-Text_Promptable_Universal_3D_Medical_Image_Segmentation_CVPR_2026_paper.html) — Maximilian Rokuss et al. | CVPR 2026 | Volumetric vision-language foundation segmentation; multi-stage text/3D feature fusion | GC;CG | [Link](https://github.com/MIC-DKFZ/VoxTell) |
| P26-04 | [HEDS-Net: A Hybrid State-Space Architecture with Axial Bridge and Progressive Weighting for Medical Image Segmentation](https://papers.miccai.org/miccai-2026/0452-Paper4167.html) — Jiaying Fan et al. | MICCAI 2026 | Hybrid vision SSM; axial cross-information skip bridge; progressive deep supervision | LP;GC;CE | [Link](https://github.com/Momentisbeused/HEDS-Net) |
| P26-05 | [VM-NeXT UNet: Synergizing ConvNeXT and Visual Mamba for Robust Medical Image Segmentation](https://papers.miccai.org/miccai-2026/1140-Paper4748.html) — Boxuan Cao and Peilin Jiang | MICCAI 2026 | Parallel ConvNeXT and visual-SSM encoders; channel–spatial skip gating; multi-scale decoder | LP;GC;CE | Not confirmed |
| P26-06 | [Glasses: Adapter-Enhanced DINOv3 with Boundary-Aware Multi-Scale Representations for Medical Image Segmentation](https://papers.miccai.org/miccai-2026/0429-Paper1781.html) — Jiahao Zhu et al. | MICCAI 2026 | DINOv3 adapter; multi-scale deformable attention; independent edge-aware convolutional encoder | LP;GC | [Link](https://github.com/friay/Glasses) |
| P26-07 | [Segment Anything Model with Symmetric Spectral Uncertainty Gating for Generalizable Medical Image Segmentation](https://papers.miccai.org/miccai-2026/0934-Paper4696.html) — He Li et al. | MICCAI 2026 | SAM adaptation; spectral manifold extrapolation; uncertainty-aware channel gating | LP;CG | [Link](https://github.com/HiLab-git/SSUG) |
| P26-08 | [Uncertainty-Aware Spatio-Semantic Contextual Prompts for Multimodal Medical Segmentation](https://papers.miccai.org/miccai-2026/1102-Paper0892.html) — Soumitri Chattopadhyay, Başar Demir, and Marc Niethammer | MICCAI 2026 | Frozen SAM-Med3D; spatial registration and semantic priors; confidence-gated dense prompting | LP;CG | [Link](https://github.com/ucsdbiag/spatio-semantic-in-context-segmentation) |
| P26-09 | [Modality-Agnostic Domain Generalizable Medical Image Segmentation via Self-Coordinated Focusing](https://papers.miccai.org/miccai-2026/0666-Paper1055.html) — Wentao Tang et al. | MICCAI 2026 | Dynamic focusing convolution; interactive attention fusion decoder | LP;CG | Not confirmed |
| P26-10 | [BDFR-Net: Boundary-Aware Dual-Domain Feature Reconstruction Network for Fetal Ultrasound Image Segmentation](https://papers.miccai.org/miccai-2026/0092-Paper3612.html) — Wenfeng Zhang et al. | MICCAI 2026 | Frequency–spatial feature reconstruction; adaptive skip alignment for ultrasound boundaries | LP;CG | [Link](https://github.com/CodeCQ789/BDFR-Net) |
| P26-11 | [Dual-Adaptive SAM3: Hierarchical Routing over Low-Rank Expert Layers for Parameter-Efficient Medical Image Segmentation](https://papers.miccai.org/miccai-2026/0308-Paper0711.html) — Ying Chen et al. | MICCAI 2026 | SAM3 adaptation; task-aware sparse expert routing; shared-base low-rank experts | CE;CG | Not confirmed |
| P26-12 | [Conformal 3D Lesion Segmentation with Balanced Risk Control](https://papers.miccai.org/miccai-2026/0194-Paper2825.html) — Binyu Tan et al. | MICCAI 2026 | Model-agnostic conformal post-processing; false-negative risk control | LP;clinical-risk evaluation | [Link](https://github.com/binyutan/CLS) |

## Why each work is relevant and what would discriminate its proposed mechanism

### P26-01 — PMRNet: Physics-informed Multi-scale Refinement Network for Medical Image Segmentation

**Design:** Combines a boundary-sensitive encoder, downsampling intended to preserve information, a diffusion-based context module, and a boundary-focused decoder.

**Suggested controlled contrast:** Compare against a conventional multi-scale CNN with matched input geometry, whole-model latency, memory, and small-structure/surface metrics.

**Review status:** Before review cutoff; not in frozen 95-family inventory. [Official publication page](https://openaccess.thecvf.com/content/CVPR2026/html/Kang_PMRNet_Physics-informed_Multi-scale_Refinement_Network_for_Medical_Image_Segmentation_CVPR_2026_paper.html).

### P26-02 — MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation

**Design:** Adapts patch-level CLIP features through probabilistic image–text interaction and a soft contrastive objective; studies predictive uncertainty and transfer.

**Suggested controlled contrast:** Hold the text-prompt set and pretraining fixed; compare probabilistic attention with deterministic attention across label fractions and untouched test domains.

**Review status:** Before review cutoff; not in frozen 95-family inventory. [Official publication page](https://openaccess.thecvf.com/content/CVPR2026/html/Koleilat_MedCLIPSeg_Probabilistic_Vision-Language_Adaptation_for_Data-Efficient_and_Generalizable_Medical_Image_CVPR_2026_paper.html).

### P26-03 — VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation

**Design:** Produces 3D masks from free-form text descriptions using multi-stage fusion of volumetric features and language representations.

**Suggested controlled contrast:** Evaluate unseen anatomical concepts and acquisition sites with a fixed prompt-information budget; report prompting burden and volumetric consistency.

**Review status:** Before review cutoff; not in frozen 95-family inventory. [Official publication page](https://openaccess.thecvf.com/content/CVPR2026/html/Rokuss_VoxTell_Free-Text_Promptable_Universal_3D_Medical_Image_Segmentation_CVPR_2026_paper.html).

### P26-04 — HEDS-Net: A Hybrid State-Space Architecture with Axial Bridge and Progressive Weighting for Medical Image Segmentation

**Design:** Combines sequential context propagation with multi-scale enhancement, an axial bridge across features, and dynamically weighted auxiliary supervision.

**Suggested controlled contrast:** Ablate the bridge, state-space path, and deep-supervision weighting independently; compare surface error and measured inference cost.

**Review status:** After 2026-06-30 review cutoff. [Official publication page](https://papers.miccai.org/miccai-2026/0452-Paper4167.html).

### P26-05 — VM-NeXT UNet: Synergizing ConvNeXT and Visual Mamba for Robust Medical Image Segmentation

**Design:** Integrates CNN local feature extraction and visual state propagation with input-adaptive skip transfer and multi-scale reconstruction.

**Suggested controlled contrast:** Use matched encoder/decoder capacity to isolate the contribution of dual encoding, gating, and loss design; measure boundary fidelity and latency.

**Review status:** After 2026-06-30 review cutoff. [Official publication page](https://papers.miccai.org/miccai-2026/1140-Paper4748.html).

### P26-06 — Glasses: Adapter-Enhanced DINOv3 with Boundary-Aware Multi-Scale Representations for Medical Image Segmentation

**Design:** Adapts pretrained DINOv3 representations through multi-scale structural priors while learning explicit boundary cues in a complementary CNN branch.

**Suggested controlled contrast:** Hold DINOv3 initialization constant and contrast adapter-only, boundary-branch-only, and combined designs under matched resource budgets.

**Review status:** After 2026-06-30 review cutoff. [Official publication page](https://papers.miccai.org/miccai-2026/0429-Paper1781.html).

### P26-07 — Segment Anything Model with Symmetric Spectral Uncertainty Gating for Generalizable Medical Image Segmentation

**Design:** Combines spectral style perturbation with uncertainty-guided channel calibration and hierarchical consistency for single-source domain generalization.

**Suggested controlled contrast:** Separate spectral augmentation from gating and consistency; evaluate unseen institutions with matched preprocessing and calibration.

**Review status:** After 2026-06-30 review cutoff. [Official publication page](https://papers.miccai.org/miccai-2026/0934-Paper4696.html).

### P26-08 — Uncertainty-Aware Spatio-Semantic Contextual Prompts for Multimodal Medical Segmentation

**Design:** Fuses registration-derived spatial context and semantic feature matching into a dense prompt for a frozen volumetric foundation model.

**Suggested controlled contrast:** Use the same support cases and target-label ontology to compare spatial-only, semantic-only, and uncertainty-gated fusion across modalities.

**Review status:** After 2026-06-30 review cutoff. [Official publication page](https://papers.miccai.org/miccai-2026/1102-Paper0892.html).

### P26-09 — Modality-Agnostic Domain Generalizable Medical Image Segmentation via Self-Coordinated Focusing

**Design:** Uses target-aware dynamic receptive-field selection and interaction across decoder semantic hierarchies to address cross-modality changes.

**Suggested controlled contrast:** Evaluate untouched hospitals/modalities with a common label space; separately ablate dynamic convolution and interactive fusion.

**Review status:** After 2026-06-30 review cutoff. [Official publication page](https://papers.miccai.org/miccai-2026/0666-Paper1055.html).

### P26-10 — BDFR-Net: Boundary-Aware Dual-Domain Feature Reconstruction Network for Fetal Ultrasound Image Segmentation

**Design:** Combines frequency-decoupled edge features with spatial context and attention-based cross-layer alignment for low-contrast fetal ultrasound.

**Suggested controlled contrast:** Factorially compare frequency features, spatial boundary supervision, and skip alignment; report noise sensitivity and independent-center transfer.

**Review status:** After 2026-06-30 review cutoff. [Official publication page](https://papers.miccai.org/miccai-2026/0092-Paper3612.html).

### P26-11 — Dual-Adaptive SAM3: Hierarchical Routing over Low-Rank Expert Layers for Parameter-Efficient Medical Image Segmentation

**Design:** Uses joint visual/text routing to select low-rank expert updates on a shared pretrained backbone; the paper reports parameter and FLOP reductions.

**Suggested controlled contrast:** Measure full inference latency and memory, router utilization, and held-out domain transfer versus matched PEFT and MoE baselines.

**Review status:** After 2026-06-30 review cutoff. [Official publication page](https://papers.miccai.org/miccai-2026/0308-Paper0711.html).

### P26-12 — Conformal 3D Lesion Segmentation with Balanced Risk Control

**Design:** Calibrates segmentation thresholds on held-out data to control lesion false-negative risk under exchangeability; this is a post-processor, not a new network architecture.

**Suggested controlled contrast:** Report per-case coverage validity, false positives, and reliability under distribution shift with independent calibration/test partitions.

**Review status:** After 2026-06-30 review cutoff; contextual evaluation method only. [Official publication page](https://papers.miccai.org/miccai-2026/0194-Paper2825.html).

## Evidence and maintenance policy

- Publication metadata and design descriptions were checked against the linked proceedings pages on 19 November 2026; the technical summaries paraphrase the authors' stated mechanisms.
- Reported performance claims are **author-reported**, not independently reproduced. Do not combine numeric values across studies with incompatible data splits, input geometry, label spaces, prompt budgets, or metric conventions.
- A code link appears only where a project URL was supplied by the original paper's publication page or its paper description; no guarantee is made about reproducibility or maintenance of the linked code.
- For a future manuscript update, review each study against the frozen corpus eligibility policy, establish bibliographic and configuration provenance, and record any new numeric rows in a separately versioned audit.
- The conformal lesion segmentation entry is **evaluation/post-processing context**, not a counted architectural method.
- Papers are linked; copyrighted PDF files are **not** mirrored in this repository.
