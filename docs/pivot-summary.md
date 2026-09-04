# DeepTrace — the pivot, in plain language (2026-09-03)

## 0. The framing we're using (tightened 2026-09-03)

**Premise (conservative — defensible):**
> Practical deepfake detection systems must balance detection performance against
> computational cost and inference latency, while remaining robust to compression
> and other post-processing commonly encountered in real-world media. One
> architectural choice is whether a conventional spatial CNN benefits sufficiently
> from an explicit frequency-domain branch to justify its additional computational
> cost.

**Central research question:**
> Under constrained inference budgets and increasing levels of video/image
> compression, does adding an explicit frequency-domain branch to a spatial CNN
> provide enough robustness improvement to justify its computational overhead?

This is a **trade-off study**, not a "we built a better detector" paper and not a
"frequency is useless" debunking. Whatever the c40 result is, we get a clean answer
(e.g. "worth it above compression level X, not below"). The negative findings
(F3-Net's c23 advantage doesn't hold on a matched backbone; the fusion gate never
engages; the spectral signal survives compression but is non-discriminative — CKA
shows it's representationally *distinct* from the spatial branch, not redundant
with it, just not useful) are **supporting evidence** for the trade-off answer, not
standalone claims.

**Do NOT claim** knowledge of how proprietary platform detectors are architected or
deployed — the literature establishes that latency/compute/compression are
important *conditions*, not that we know Meta's stack.

## 1. What we set out to do

Build a deepfake detector that works by **frequency analysis**. Think of it like a
graphic equaliser for images: instead of looking at the picture directly, you break
it into "coarse" patterns (big smooth shapes) and "fine" patterns (sharp edges,
tiny textures). The theory: the AI tools that make deepfakes leave faint, regular
"fingerprints" in the fine patterns that a normal image classifier might miss.

Our version added a **custom DCT layer** (DCT = the maths that turns an image into
those coarse-to-fine frequency components — the same transform inside JPEG) that the
network could *tune while it learned*, plus a few other ideas (a learnable "mask"
over frequencies, random frequency drop-out, and fake-artifact simulation).

**The bar for success:** beat a well-known published benchmark from a top-tier
conference. In research, "we built a thing" isn't a paper — "we built a thing that
measurably beats what the field already has" is.

## 2. What we found — and why we pivoted

We ran the full comparison properly: our method vs. strong ordinary image
classifiers (the "spatial" baselines, e.g. Xception), each trained **3 times** with
different random starts so we could tell real differences from luck.

Result: **our frequency approach did not beat the ordinary classifier.**

- Head-to-head it *tied at best and lost at worst.*
- The pure frequency-only model scored 0.70 AUC — barely above guessing on the hard
  cases (1.0 = perfect, 0.5 = coin flip).
- Every extra idea we added (mask, drop-out, simulation) changed the score by
  nothing measurable; a couple made it slightly worse.

So the "we beat the benchmark" paper was off the table. Rather than abandon the
work, we **pivoted the framing**:

> **From:** "Here is a better deepfake detector."
> **To:** "Here is a careful, fair test showing that adding frequency analysis does
> *not* beat a well-tuned ordinary CNN for deepfake detection — and here is why."

This is called a **controlled analysis** or **negative-results** paper. It's a
legitimate, publishable contribution.

## 3. Why the pivot matters (the significance)

- Frequency-based deepfake detection is a **whole sub-field** built on 2020-era
  papers. Many later papers just *assume* it helps.
- Methods often look great in their original paper but don't hold up when someone
  re-tests them fairly (same backbone, same data budget, same training tricks,
  multiple random seeds). Our study is exactly that fair re-test.
- A clear result — "under matched conditions the frequency advantage largely
  disappears" — **saves other researchers wasted effort** and is the kind of
  rigorous, honest work that media-forensics reviewers value.
- We're not fudging anything to claim a win. The contribution *is* the careful
  measurement.

## 4. Results so far (plain language)

| Test | What it checks | What we found |
|---|---|---|
| **In-domain** | Detect fakes of the same kind you trained on | Our method ≈ ordinary CNN. Frequency-only ≈ near chance (0.70). |
| **Compression robustness** | The frequency camp's strongest claim: "frequency survives when the video is squashed" | Re-tested with heavy JPEG. The frequency model's apparent edge was a **fluke of one random seed** — it vanished once we averaged 3 runs. Our full method was actually *worse* under compression. |
| **Which fake types** | Break the score down by the 4 deepfake methods in the data | On the **subtlest** fake type (NeuralTextures), the plain CNN was **best**. The frequency-only model did best on the **crudest** fakes — the *opposite* of what the theory predicts. |
| **The "frequency dial"** | Our model has a learnable knob balancing frequency vs. ordinary features | The knob **never moved** from its 50/50 start — **but** (added 2026-09-04) we found the training setup was accidentally nudging that knob back toward 50/50 on every step regardless of what it "wanted" to do (a technical bug: a setting called weight decay, meant for other parts of the model, was leaking onto this knob too). Fixed in code; needs a re-run to confirm the "no reason to lean on frequency" reading survives once the bug is gone. Flagging honestly rather than overselling it. |
| **Spectrum analysis** | Is there *any* real real-vs-fake difference in the frequency domain? | Yes — but **tiny** (a small statistical effect), and heavy compression barely dents it. So frequency doesn't fail because "compression destroys the signal" — it fails because the signal is **weak and adds little on top of what the ordinary CNN already predicts.** |

**The refined takeaway (tightened 2026-09-04):** a real but faint frequency
fingerprint exists; it's compression-resistant; and it has **low incremental
predictive value** on top of an ordinary spatial CNN — meaning it doesn't move the
accuracy needle, not (yet) that the CNN literally "sees" the same thing internally;
that stronger claim needs one more check we haven't run. A dedicated frequency
pathway adds nothing measurable and can hurt generalisation to other datasets.

## 5. The base paper, the recent target, and where to find them

*(Updated 2026-09-04 — the base/target pairing was sharpened, see below for why.)*

**Base / foundation paper** (the method we actually build and directly test):

> **"Thinking in Frequency: Face Forgery Detection by Mining Frequency-Aware Clues"**
> — Yuyang Qian, Guojun Yin, Lu Sheng, Zixuan Chen, Jing Shao. Often called
> **F3-Net**. **ECCV 2020** — the *European Conference on Computer Vision*, one of
> the top three computer-vision conferences (with CVPR and ICCV).

**How to find it:** search `F3-Net Thinking in Frequency ECCV 2020`; free PDF on
arXiv and the ECCV 2020 open-access proceedings (SpringerLink, LNCS 12357); code
(unofficial) at github.com/yyk-wew/F3Net.

This is our base paper because we don't just cite it — we **built its FAD
architecture into our own code** and are running the one experiment (the c40 run,
see §7/§8) that tests its headline claim directly: that its frequency features help
**most on heavily-compressed video**.

**Recent target paper** (the current state of the art we position against):

> **"FreqDebias: Towards Generalizable Deepfake Detection via Consistency-Driven
> Frequency Debiasing"** — Hossein Kashiani, Niloufar Alipour Talemi, Fatemeh
> Afghah. **CVPR 2025**, pp. 8775–8785. arXiv:2509.22412.

**Why this is the right recent paper — a CONTRAST, not an ally (corrected
2026-09-04):** five years after F3-Net, this is the most current major work in
frequency-domain face-forgery detection. It names a new problem — **"spectral
bias"**: detectors that over-rely on specific frequency bands generalise *worse* to
unseen forgeries — and proposes a training-time fix (augmentation + consistency
regularisation) so the model can keep using frequency safely. **This is not the
same conclusion as ours.** FreqDebias's position is "frequency helps, if you
correct for its failure mode"; our (still provisional) position is "a simple,
matched-backbone frequency branch adds nothing, with no special machinery." The
useful framing: FreqDebias's existence and complexity is itself evidence that
*naive* frequency modelling (like F3-Net's FAD, like our own branch) doesn't just
work — real engineering was needed to make theirs pay off. We cite and position
against it rather than fully reproducing its (heavier) training method — but we
borrow its diagnostic idea cheaply: our existing frequency-band analysis
(`results/analysis/spectra/`) can directly check
whether *our* frequency branch shows the same "spectral bias" they describe.

Antecedents (the older foundation both F3-Net and FreqDebias build on) and the full
supporting-paper list are in `docs/related-work.md` (Frank et al. ICML 2020; Zhang
et al. WIFS 2019; Durall et al. CVPR 2020; SBI CVPR 2022; Ojha et al. CVPR 2023;
Gragnaniello et al. ICME 2021; Corvi et al. ICASSP 2023).

## 6. Where the results live

Everything is committed to the repo under `results/` (permanent, version-controlled):
`results/README.md` and `results/analysis/README.md` have the full tables.
Local backups: `~/Downloads/deeptrace_results_20260904.tar.gz` and
`~/Downloads/analysis_final_0117.tar.gz`.

## 7. Real-world application (why anyone should care)

This is a **"science that guides practice"** paper — it changes what people build
and where they spend effort, not a product.

**Who acts differently because of it:**

- **Teams that deploy deepfake detectors** — platform moderation (Meta, TikTok,
  YouTube), newsroom verification (Reuters, AFP), identity-verification vendors
  (Onfido, Jumio, bank video-KYC), video-call security. They face a concrete design
  choice: bolt a frequency-analysis branch onto the detector? It costs latency,
  memory, complexity. Our result: on a matched backbone it is not worth it — put
  the compute into the spatial model, data, and augmentation.
- **At platform scale** (billions of items scanned) a frequency branch that ~doubles
  inference FLOPs for zero accuracy gain is real money and energy. This is the
  efficiency angle.
- **Practitioner robustness expectations** — deployed content is always recompressed
  on upload; someone reading F3-Net might over-trust frequency methods for
  compressed video. Our compression results correct that.
- **The research community** — frequency-domain deepfake detection still gets new
  papers (2024). A rigorous "here is exactly what it buys you and where the
  boundary is" redirects effort toward what generalises (semantic / CLIP features,
  self-blending).
- **Reusable methodology** — the controlled protocol (matched backbone, seeds, the
  fusion-gate readout that shows a component "doesn't engage") is a template for
  fairly evaluating any "add-a-branch" claim in detection.

**Bigger-picture stakes (for the intro):** deepfake detection matters because of
non-consensual intimate imagery (most real-world deepfake harm), video/voice fraud
(2024 Arup Hong Kong case — $25M lost to a deepfake video call), identity-
verification bypass, and election/disinformation. Detectors are the defensive
layer; this paper makes them cheaper and better-targeted.

**Draft application paragraph (conservative):**

> Practical deepfake detection systems balance detection performance against
> computational cost and inference latency while remaining robust to compression and
> post-processing common in real-world media. A recurring architectural choice is
> whether a spatial CNN benefits enough from an explicit frequency-domain branch to
> justify its overhead. We study this trade-off under matched training and
> increasing compression, and find the frequency branch adds cost without a
> commensurate robustness or accuracy gain across the regimes we test — evidence
> that, for this class of face-forgery detection, budget is better spent on the
> spatial model, data, and augmentation.

**Related design pattern (active — verify authors/venues before citing):** dual-domain /
frequency-aware detectors combining spatial features with DCT/wavelet/frequency
representations — e.g. TSFF-Net, WGN, DSTF-Net, FreqMamba, hybrid spatial-frequency
EfficientNet. Establishes the design choice is current, not a 2020 relic.

## 8. What's left before writing the paper

1. **The c40 experiment** — train *and* test at heavy compression (F3-Net's home
   turf). ~5 hours of Kaggle GPU when the quota resets. Plan in
   `docs/phase3-plan.md`.
2. **Cross-dataset error bars** — repeat the "train here, test on a different
   deepfake dataset" check for 2 more random seeds (~35 min).
3. **Write-up** — the numbers are essentially in; draft against
   `docs/phase3-plan.md` and `docs/related-work.md`.
