# Related work — base paper, target, and supporting literature

Positioning for the controlled-study / trade-off paper.
**Central research question:** under constrained inference budgets and increasing
compression, does adding an explicit frequency-domain branch to a spatial CNN
provide enough robustness improvement to justify its computational overhead?

All citations below were verified against dblp / CVF / MLR / arXiv on 2026-09-03/04.
Re-verify before final submission.

---

## Base / foundation paper (the method we implement and directly engage)

**Qian, Yin, Sheng, Chen, Shao. "Thinking in Frequency: Face Forgery Detection by
Mining Frequency-Aware Clues (F3-Net)." ECCV 2020.**

Two frequency streams (FAD: frequency-aware decomposition; LFS: local frequency
statistics) on an Xception backbone. Headline claim: beats spatial baselines on
FaceForensics++ **at every compression level, with the largest lead on low-quality
(c40) media**. This is *our* base paper because we don't just cite it — we
**implement its FAD architecture** (`src/models/f3net.py`) and **directly test its
central claim** with a matched-backbone c40 training run (C7 in
`docs/validation-plan.md`). Everything in our study is built to engage this one
paper's method and this one paper's headline result.

Antecedents (the deeper premise F3-Net itself rests on — cite, don't reproduce):

- **Frank, Eisenhofer, Schönherr, Fischer, Kolossa, Holz. "Leveraging Frequency
  Analysis for Deep Fake Image Recognition." ICML 2020.** arXiv:2003.08685 · code:
  RUB-SysSec/GANDCTAnalysis. Established DCT-domain analysis — the transform in our
  custom layer — as a detection tool; GAN images carry a regular grid artifact in
  the DCT spectrum. Our learnable per-coefficient DCT mask is a parametrised
  generalisation of their *fixed* DCT analysis.
- **Zhang, Karaman, Chang. "Detecting and Simulating Artifacts in GAN Fake Images."
  WIFS 2019.** arXiv:1907.06515. Up-sampling produces *spectral replication*
  artifacts; introduces the **AutoGAN** simulator — the direct ancestor of our
  Spectral Artifact Simulation (SAS).
- **Durall, Keuper, Keuper. "Watch Your Up-Convolution: CNN Based Generative Deep
  Neural Networks Are Failing to Reproduce Spectral Distributions." CVPR 2020.**
  arXiv:2003.01826. A spectral classifier detects the up-convolution mismatch "with
  up to 100% accuracy" on public benchmarks — the strong claim our data pushes back
  on.
- **Odena, Dumoulin, Olah. "Deconvolution and Checkerboard Artifacts." Distill,
  2016.** The mechanism: fixed-stride transposed conv leaves periodic checkerboard
  patterns.

---

## Recent target (the current state of the art we position against)

**Kashiani, Alipour Talemi, Afghah. "FreqDebias: Towards Generalizable Deepfake
Detection via Consistency-Driven Frequency Debiasing." CVPR 2025.**
pp. 8775–8785 · arXiv:2509.22412 ·
[CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Kashiani_FreqDebias_Towards_Generalizable_Deepfake_Detection_via_Consistency-Driven_Frequency_Debiasing_CVPR_2025_paper.html)

Defines **spectral bias**: detectors over-rely on specific frequency bands, which
*restricts* generalisation to unseen forgeries — a new failure mode diagnosed five
years after F3-Net. Fixes it with **Forgery Mixup (Fo-Mixup)** augmentation that
diversifies training-sample frequency characteristics, plus a **dual consistency
regulariser** (local, via class-activation maps; global, via a von Mises-Fisher
distribution over embeddings).

Why it's the right recent target: it is the most current major statement in
frequency-domain face-forgery detection (closes the base-paper-recency gap — F3-Net
2020 → FreqDebias 2025), and its own diagnosis is **independent, top-venue support
for our scepticism about frequency reliance** — it just responds by debiasing rather
than dropping the frequency pathway. Their result also implies that *unregularised*
frequency reliance (which is what F3-Net's FAD, and our custom DCT branch, are) is
liable to exactly the failure our study documents.

**Scope note — do not fully reproduce.** Fo-Mixup + dual CR is materially heavier to
train than F3-Net's FAD. Engage it via (a) citation/positioning, and (b) their own
diagnostic lens applied cheaply to our data: the DCT band-ablation and
per-coefficient t-map we already compute (`src/spectra.py`, `results/analysis/`) is
a direct test of whether *our* frequency branch exhibits the spectral bias they
describe — same vocabulary, no need to reproduce their training pipeline.

Secondary/contemporaneous targets (frequency-domain face-forgery detectors, for the
"the design pattern is current, not a 2020 relic" argument — verify before citing):

- **Liu et al. "Spatial-Phase Shallow Learning (SPSL)." CVPR 2021.** Phase spectrum
  + shallow net.
- **Li et al. "Frequency-aware Discriminative Feature Learning (FDFL)." CVPR 2021.**
- **Doloriel, Cheung. "Frequency Masking for Universal Deepfake Detection."
  ICASSP 2024.** arXiv:2401.06506.
- TSFF-Net, WGN, DSTF-Net, FreqMamba, hybrid spatial-frequency EfficientNet —
  recent dual-domain architectures (unverified, confirm authors/venues before use).

---

## Supporting papers (independent evidence for the claim)

### A. Spatial / semantic representations generalise better than spectral ones

- **Shiohara, Yamasaki. "Detecting Deepfakes with Self-Blended Images (SBI)."
  CVPR 2022 (oral).** arXiv:2204.08376. ~93% AUC on Celeb-DF with **no real
  deepfakes in training** — a purely spatial blending-artifact method that is still
  among the strongest cross-dataset results.
- **Ojha, Li, Lee. "Towards Universal Fake Image Detectors that Generalize Across
  Generative Models." CVPR 2023.** arXiv:2302.10174. A *frozen* CLIP (semantic,
  non-spectral) feature space with nearest-neighbours generalises far better than
  trained spectral/artifact features (+15 mAP, +26% acc on unseen models).
- **Dong et al. "Implicit Identity Leakage (CADDM)." CVPR 2023.** Spatial
  artifact-detection module; strong cross-dataset.
- **Yan et al. "UCF: Uncovering Common Features for Generalizable Deepfake
  Detection." ICCV 2023.** Spatial-domain disentanglement of content vs forgery.

### B. Frequency cues are fragile / biased — compression, post-processing, transfer

- **FreqDebias (see above)** — spectral bias as an intrinsic liability of
  unregularised frequency reliance.
- **Gragnaniello, Cozzolino, Marra, Poggi, Verdoliva. "Are GAN Generated Images
  Easy to Detect? A Critical Analysis of the State-of-the-Art." ICME 2021.**
  arXiv:2104.02617. Detectors — frequency ones included — collapse under realistic
  conditions. Methodological precedent for a controlled critical re-evaluation.
- **Corvi, Cozzolino, Zingarini, Poggi, Nagano, Verdoliva. "On the Detection of
  Synthetic Images Generated by Diffusion Models." ICASSP 2023.** GAN-era spectral
  fingerprints do **not** transfer to diffusion images — generator-specific, so a
  frequency detector's inductive bias is a liability out of distribution.
- **Grommelt, Weiss, Pfreundt, Keuper. "Fake or JPEG? Revealing Common Biases in
  Generated Image Detection Datasets." 2024 (arXiv:2403.17608).** Frequency
  detectors frequently learn JPEG/compression statistics, not generation artifacts.

### C. Controlled-benchmark methodology (how to run a fair comparison)

- **Yan, Zhang, Yao, Fu, Wu, Yang, et al. "DeepfakeBench: A Comprehensive Benchmark
  of Deepfake Detection." NeurIPS 2023 (Datasets & Benchmarks).** arXiv:2307.01426.
  Standardised preprocessing, splits, and backbones; our results should track its
  leaderboard for the detectors we re-implement.
- **Yan et al. "DF40: Toward Next-Generation Deepfake Detection." NeurIPS 2024.**
  40 forgery methods incl. diffusion/editing — a modern cross-generator target.

### D. Nuance — a spatial method that exploits an upsampling (frequency) phenomenon

- **Tan, Wei, Yao, et al. "Rethinking the Up-Sampling Operations in CNN-based
  Generative Networks for Generalizable Deepfake Detection (NPR)." CVPR 2024.**
  arXiv:2312.10461. The useful signal from upsampling is captured best by a
  **pixel-domain** operator, not an explicit frequency transform — the phenomenon
  is real, the frequency *representation* is not the way to use it.

---

## One-paragraph positioning (for the intro)

> F3-Net (ECCV 2020) established that injecting an explicit frequency-domain branch
> into a spatial CNN improves face-forgery detection, with its largest reported
> gains on heavily compressed video. Five years on, the most recent major work in
> this space, FreqDebias (CVPR 2025), diagnoses a new failure mode — *spectral
> bias*, where unregularised frequency reliance itself restricts generalisation —
> and proposes consistency-driven training to correct it. We ask a narrower,
> practitioner-facing question: under a matched backbone, training budget, and
> increasing compression, does an explicit frequency branch (F3-Net's FAD, and our
> own learnable DCT front-end) earn its computational cost at all? Across in-domain,
> per-manipulation, cross-dataset, and compression-robustness evaluation, we find it
> does not, consistent with the spectral bias FreqDebias identifies and with the
> generalisation-favours-spatial pattern reported by Gragnaniello et al. (ICME
> 2021), Corvi et al. (ICASSP 2023), and Ojha et al. (CVPR 2023).
