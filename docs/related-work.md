# Related work — base paper, target, and supporting literature

Positioning for the controlled-study / analysis paper.
**Main claim:** on a matched backbone and a controlled training budget, adding
frequency-domain modelling (a learnable DCT front-end, FAD-style band filtering)
does **not** improve deepfake detection over a well-tuned spatial CNN — in-domain,
cross-dataset, or under compression.

All citations below were verified against dblp / CVF / MLR / arXiv on 2026-09-03.
Re-verify before final submission.

---

## Base paper (the foundation the whole line of work rests on)

**Frank, Eisenhofer, Schönherr, Fischer, Kolossa, Holz.
"Leveraging Frequency Analysis for Deep Fake Image Recognition." ICML 2020.**
arXiv:2003.08685 · code: RUB-SysSec/GANDCTAnalysis

Why it is *our* base paper: it established DCT-domain analysis — the exact transform
in our custom layer — as a deepfake/GAN-detection tool. It showed that GAN images
carry a regular grid artifact in the DCT spectrum and that a classifier on DCT
coefficients separates real from fake. Our learnable per-coefficient DCT mask is a
parametrised generalisation of their *fixed* DCT analysis; F3-Net's FAD is another.
The analysis paper's framing is: "the fixed-transform predecessor (Frank et al.)
and its learnable descendants (F3-Net, ours) do not beat a spatial CNN once the
backbone and data budget are matched."

Deeper antecedents (cite as the physical origin of the artifact):

- **Zhang, Karaman, Chang. "Detecting and Simulating Artifacts in GAN Fake Images."
  WIFS 2019.** arXiv:1907.06515. Up-sampling produces *spectral replication*
  artifacts; trains a spectrum-based classifier; introduces the **AutoGAN**
  simulator — the direct ancestor of our Spectral Artifact Simulation (SAS).
- **Durall, Keuper, Keuper. "Watch Your Up-Convolution: CNN Based Generative Deep
  Neural Networks Are Failing to Reproduce Spectral Distributions." CVPR 2020.**
  arXiv:2003.01826. Transposed convolution cannot match the training data's
  spectrum; a spectral classifier detects this "with up to 100% accuracy" on
  public benchmarks — the strong claim our compressed-data results push back on.
- **Odena, Dumoulin, Olah. "Deconvolution and Checkerboard Artifacts." Distill,
  2016.** The mechanism: fixed-stride transposed conv leaves periodic checkerboard
  patterns.

---

## Target paper (the benchmark the claim contests)

**Qian, Yin, Sheng, Chen, Shao. "Thinking in Frequency: Face Forgery Detection by
Mining Frequency-Aware Clues (F3-Net)." ECCV 2020.**
Two frequency streams (FAD: frequency-aware decomposition; LFS: local frequency
statistics) on an Xception backbone. Headline result: beats spatial baselines on
FaceForensics++ **at every compression level, with the largest lead on low-quality
(c40) media**. That c40 claim is precisely what our robustness sweep and a matched
c40 training run must engage — our current evidence is c23 only.

Secondary targets (frequency-domain face-forgery detectors to position against):

- **Liu et al. "Spatial-Phase Shallow Learning (SPSL): Rethinking Face Forgery
  Detection in Frequency Domain." CVPR 2021.** Phase spectrum + shallow net.
- **Li et al. "Frequency-aware Discriminative Feature Learning (FDFL)." CVPR 2021.**
- **Gu et al. / Luo et al. "Generalizing Face Forgery Detection with High-frequency
  Features." CVPR 2021.**

---

## Supporting papers (independent evidence for the claim)

### A. Spatial / semantic representations generalise better than spectral ones

- **Shiohara, Yamasaki. "Detecting Deepfakes with Self-Blended Images (SBI)."
  CVPR 2022 (oral).** arXiv:2204.08376. ~93% AUC on Celeb-DF with **no real
  deepfakes in training** — a purely spatial blending-artifact method that is still
  among the strongest cross-dataset results. Primary support: the field's best
  generalisation comes from spatial artifact modelling, not frequency.
- **Ojha, Li, Lee. "Towards Universal Fake Image Detectors that Generalize Across
  Generative Models." CVPR 2023.** arXiv:2302.10174. Training a classifier on
  spectral/artifact features overfits to the training generator; a *frozen* CLIP
  (semantic, non-spectral) feature space with nearest-neighbours generalises far
  better (+15 mAP, +26% acc on unseen diffusion/AR models). The SOTA generalisation
  recipe is explicitly *not* frequency-based.
- **Dong et al. "Implicit Identity Leakage (CADDM)." CVPR 2023.** Spatial
  artifact-detection module; strong cross-dataset.
- **Yan et al. "UCF: Uncovering Common Features for Generalizable Deepfake
  Detection." ICCV 2023.** Spatial-domain disentanglement of content vs forgery.

### B. Frequency cues are fragile — compression / post-processing / transfer

- **Gragnaniello, Cozzolino, Marra, Poggi, Verdoliva. "Are GAN Generated Images
  Easy to Detect? A Critical Analysis of the State-of-the-Art." ICME 2021.**
  arXiv:2104.02617. Detectors — frequency ones included — collapse under realistic
  conditions (social-media recompression, unseen architectures). The methodological
  precedent for our paper: a controlled critical re-evaluation.
- **Corvi, Cozzolino, Zingarini, Poggi, Nagano, Verdoliva. "On the Detection of
  Synthetic Images Generated by Diffusion Models." ICASSP 2023.**
  arXiv (grip-unina/DMimageDetection). GAN-era spectral fingerprints do **not**
  transfer to diffusion images — the spectral artifact is generator-specific, so a
  frequency detector's inductive bias is a liability out of distribution.
- **Grommelt, Weiss, Pfreundt, Keuper. "Fake or JPEG? Revealing Common Biases in
  Generated Image Detection Datasets." 2024 (arXiv:2403.17608 / ECCV-W).**
  Frequency detectors frequently learn **JPEG/compression statistics**, not
  generation artifacts, because fake/real splits differ in compression. Pairs
  directly with our "real-vs-fake DCT gap is gone at c40" figure.
- **Frank et al. (base paper) itself** reports that adding mild perturbations /
  recompression erodes the DCT-spectrum signal — usable as a self-consistent
  citation.

### C. Controlled-benchmark methodology (how to run a fair comparison)

- **Yan, Zhang, Yao, Fu, Wu, Yang, et al. "DeepfakeBench: A Comprehensive Benchmark
  of Deepfake Detection." NeurIPS 2023 (Datasets & Benchmarks).**
  arXiv:2307.01426. Standardised preprocessing, splits, and backbones; our results
  should track its leaderboard for the detectors we re-implement.
- **Yan et al. "DF40: Toward Next-Generation Deepfake Detection." NeurIPS 2024.**
  40 forgery methods incl. diffusion/editing — the modern cross-generator target
  for our Stage-6 extension.

### D. Nuance — a spatial method that exploits an upsampling (frequency) phenomenon

- **Tan, Wei, Yao, et al. "Rethinking the Up-Sampling Operations in CNN-based
  Generative Networks for Generalizable Deepfake Detection (NPR)." CVPR 2024.**
  arXiv:2312.10461. The useful signal from upsampling (the base-paper phenomenon)
  is captured best by a **pixel-domain** operator (neighbouring-pixel relationships),
  not an explicit frequency transform. Supports our framing that the frequency
  *phenomenon* is real but the frequency *representation* is not the way to use it.

---

## One-paragraph positioning (for the intro)

> Frequency-domain analysis of generative-model artifacts, established by Zhang et
> al. (WIFS 2019), Durall et al. (CVPR 2020) and Frank et al. (ICML 2020), motivated
> a family of face-forgery detectors that inject an explicit spectral transform into
> a CNN, of which F3-Net (ECCV 2020) is the most cited, reporting its largest gains
> on heavily compressed video. We revisit that premise under a controlled protocol —
> one backbone (Xception), one data budget, matched augmentation, three seeds — and
> find that a learnable DCT front-end, an FAD-style band decomposition, and spectral
> artifact simulation each fail to improve on the spatial backbone in-domain, and
> transfer worse to Celeb-DF and DFDC. Our analysis attributes this to (i) JPEG
> quantisation erasing the real-vs-fake spectral gap on the compressed media these
> systems target, and (ii) generator-specific spectral fingerprints that do not
> transfer, consistent with Gragnaniello et al. (ICME 2021), Corvi et al. (ICASSP
> 2023) and the semantic-feature generalisation of Ojha et al. (CVPR 2023).
