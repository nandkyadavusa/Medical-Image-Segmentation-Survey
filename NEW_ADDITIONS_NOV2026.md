# Curated medical image segmentation papers — November 2026 update

**Status:** Source-linked bibliography, not a numerical verification audit. The 95-method manuscript inventory (S1), 68-record numerical audit (S2), and their literature boundary of 30 June 2026 remain unchanged. MICCAI 2026 proceedings listed here postdate that cutoff. CVPR 2026 papers are additional repository coverage, not retroactively included in the manuscript inventory.

These entries were checked against the official MICCAI 2026 Open Access or CVPR 2026 Open Access paper pages on 20 November 2026. Descriptions summarize author-stated design; they are not independent replication or claims of comparable state-of-the-art performance. Code links are author-linked or author-stated; no code execution, license audit, or reproducibility check was performed. No copyrighted PDFs were mirrored.

## 1. Adaptive Frequency-Guided Parallel Mamba Network for Polyp Segmentation

**AFMamba** · Jiang et al. · MICCAI 2026 · Frequency-aware Mamba / polyp segmentation  
[Official paper](https://papers.miccai.org/miccai-2026/0032-Paper4551.html) · [Author code](https://github.com/B00MCLAP/Polyp-AFMamba)

**Architecture:** Combines frequency-domain cues and spatial structure priors with parallel Mamba pathways for polyp boundaries and domain shift.

**Audit question:** Compare cross-dataset splits, boundary metrics, and total DCT/scan inference cost.

## 2. Towards Direction-Equilibrated Segmentation in 3D Medical Images

**DEM-Mamba** · Zhu et al. · MICCAI 2026 · Volumetric anisotropy / directional Mamba  
[Official paper](https://papers.miccai.org/miccai-2026/1074-Paper1575.html) · [Author code](https://github.com/pky888/DEM-Mamba)

**Architecture:** Orientation-aware wavelet features, bidirectional Mamba scans, and conflict-based directional fusion target anisotropic 3D boundaries.

**Audit question:** Validate improvement under matched voxel spacing, HD95, memory, and physical orientation shifts.

## 3. Dino U-Net: Exploiting High-Fidelity Dense Features from Foundation Models for Medical Image Segmentation

**Dino U-Net** · Li et al. · MICCAI 2026 · Frozen DINOv3 backbone / medical segmentation  
[Official paper](https://papers.miccai.org/miccai-2026/0281-Paper0995.html) · [Author code](https://github.com/yifangao112/DinoUNet)

**Architecture:** Uses frozen DINOv3 features with dual-branch adaptation and fidelity-aware projections to preserve dense representations.

**Audit question:** Measure backbone-size, adaptation cost, pretraining provenance, and cross-modality transfer.

## 4. Hybrid Transformer-Mamba for Weakly Supervised Volumetric Medical Segmentation

**TranSamba** · Lyu et al. · MICCAI 2026 · Weakly supervised 3D Transformer–Mamba  
[Official paper](https://papers.miccai.org/miccai-2026/0479-Paper4858.html) · [Author code](https://github.com/YihengLyu/TranSamba)

**Architecture:** Cross-plane Mamba adds inter-slice exchange to a ViT backbone under plane-level weak supervision.

**Audit question:** Inspect weak-label budgets, depth-dependent memory, and cross-plane leakage prevention.

## 5. SRMA-Mamba: Spatial Reverse Mamba Attention Network for Pathological Liver Segmentation in MRI Volumes

**SRMA-Mamba** · Zeng et al. · MICCAI 2026 · 3D MRI / anatomy-aware Mamba  
[Official paper](https://papers.miccai.org/miccai-2026/0992-Paper5682.html) · [Author code](https://github.com/JunZengz/SRMA-Mamba)

**Architecture:** Anatomical multi-plane selective scanning and reverse-attention boundary refinement for pathological liver MRI.

**Audit question:** Check external cohort transfer and compute cost of multiplanar inference.

## 6. APEX-SAM: Anatomy-Aware Prompting with Expert Retrieval for Training-Free Medical Image Segmentation

**APEX-SAM** · Mao et al. · MICCAI 2026 · Training-free promptable foundation model  
[Official paper](https://papers.miccai.org/miccai-2026/0067-Paper0425.html) · [Author code](https://github.com/Trump0412/APEX-SAM)

**Architecture:** Uses quality-aware expert retrieval, anatomy-aware prompt mining, and multimodal prompt fusion without target-domain gradient updates.

**Audit question:** Audit expert-bank construction, support/query patient separation, retrieval cost, and label availability.

## 7. PlaneCycle: Training-Free 2D-to-3D Lifting of Foundation Models Without Adapters

**PlaneCycle** · Yu et al. · MICCAI 2026 · 2D-to-3D pretrained feature transfer  
[Official paper](https://papers.miccai.org/miccai-2026/0794-Paper0966.html) · [Author code](https://github.com/HINTLab/PlaneCycle)

**Architecture:** Cyclic orthogonal-plane feature aggregation reuses pretrained 2D representations for volumetric tasks without architectural adapters.

**Audit question:** Disentangle no-weight-update lifting from later linear probing and fine-tuning; evaluate anisotropic volumes.

## 8. Rethinking Box Supervision: Bias-Free Weakly Supervised Medical Segmentation

**WeakMed** · Wei and Huang · CVPR 2026 · Weak-label supervision  
[Official paper](https://openaccess.thecvf.com/content/CVPR2026/html/Wei_Rethinking_Box_Supervision_Bias-Free_Weakly_Supervised_Medical_Segmentation_CVPR_2026_paper.html) · Code not independently confirmed

**Architecture:** Mask-to-box training alignment and scale consistency reduce supervision bias from bounding-box annotations without extra inference modules.

**Audit question:** Compare equal annotation budgets and sensitivity to coarse or incomplete boxes.

## 9. SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation

**SD-FSMIS** · Li et al. · CVPR 2026 · Few-shot / diffusion prior  
[Official paper](https://openaccess.thecvf.com/content/CVPR2026/html/Li_SD-FSMIS_Adapting_Stable_Diffusion_for_Few-Shot_Medical_Image_Segmentation_CVPR_2026_paper.html) · Code not independently confirmed

**Architecture:** Adapts pretrained diffusion representations for few-shot segmentation using support–query interaction and visual-to-text conditioning.

**Audit question:** Measure conditioning inference overhead, sensitivity to support sets, and domain-specific pretraining effects.

## 10. From Infusion to Assimilation Distillation for Medical Image Segmentation

**IAD** · Hong et al. · CVPR 2026 · Foundation-model distillation / efficiency  
[Official paper](https://openaccess.thecvf.com/content/CVPR2026/html/Hong_From_Infusion_to_Assimilation_Distillation_for_Medical_Image_Segmentation_CVPR_2026_paper.html) · [Author code](https://github.com/hjklearn/IAD)

**Architecture:** Combines teacher–student semantic alignment with student-oriented contrastive assimilation for efficient medical segmentation.

**Audit question:** Assess teacher pretraining, student size, inference-only cost, and out-of-domain behavior.

## 11. CG-Reasoner: Centroid-Guided Positional Reasoning Segmentation for Medical Imaging with a Robust Visual-Text Consistency Metric

**CG-Reasoner** · Polamreddy and Ma · CVPR 2026 · Vision-language / geometric reasoning  
[Official paper](https://openaccess.thecvf.com/content/CVPR2026/html/Polamreddy_CG-Reasoner_Centroid-Guided_Positional_Reasoning_Segmentation_for_Medical_Imaging_with_a_CVPR_2026_paper.html) · [Author code](https://github.com/lpmm2025/CG-Reasoner)

**Architecture:** Jointly predicts segmentation and spatial lesion-centroid reasoning, proposing a metric for text–mask alignment.

**Audit question:** Test whether reasoning improves mask quality independently of language priors and positional labels.

## Methodological boundaries

- The paper list is a **non-exhaustive curated update**, not a PRISMA/systematic review selection.
- Venue and author identity were checked from official conference pages; results, latency, parameter counts and test splits were **not numerically transcribed or verified** for this batch.
- New entries must not be added to the S1 numerator (95) without re-checking duplicate method families, inclusion criteria, manuscript citations, and the associated S2 evidence trail.
- For repository maintenance, review new citations against the root README and earlier `ADDITIONAL_PAPERS_2026.md` before merging.
