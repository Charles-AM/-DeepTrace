# DeepTrace — the pivot, in plain language (2026-09-03)

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
| **The "frequency dial"** | Our model has a learnable knob balancing frequency vs. ordinary features | The knob **never moved** from its 50/50 start. The model saw no reason to lean on frequency. |
| **Spectrum analysis** | Is there *any* real real-vs-fake difference in the frequency domain? | Yes — but **tiny** (a small statistical effect), and heavy compression barely dents it. So frequency doesn't fail because "compression destroys the signal" — it fails because the signal is **weak and the ordinary CNN already picks it up.** |

**The refined takeaway:** a real but faint frequency fingerprint exists; it's
compression-resistant; and an ordinary spatial CNN already captures whatever
usefulness it has. A dedicated frequency pathway adds nothing and can hurt
generalisation to other datasets.

## 5. The target paper and where to find it

**Target paper** (the benchmark our study argues against):

> **"Thinking in Frequency: Face Forgery Detection by Mining Frequency-Aware Clues"**
> — Yuyang Qian, Guojun Yin, Lu Sheng, Zixuan Chen, Jing Shao.
> Often called **F3-Net**.

**Conference:** **ECCV 2020** — the *European Conference on Computer Vision*, one of
the top three computer-vision conferences in the world (with CVPR and ICCV).

**How to find it:**
- Search: `F3-Net Thinking in Frequency ECCV 2020`
- Free PDF: arXiv (search the title) and the ECCV 2020 open-access proceedings
  (`eccv2020.eu` / SpringerLink, vol. LNCS 12357)
- Code (unofficial): github.com/yyk-wew/F3Net

F3-Net's headline claim is that its frequency features help **most on
heavily-compressed video** (the "c40" setting). That specific claim is the one
experiment we still owe (see §7).

**Base paper** (the older foundation the whole idea — ours and F3-Net's — rests on):

> **"Leveraging Frequency Analysis for Deep Fake Image Recognition"**
> — Joel Frank, Thorsten Eisenhofer, Lea Schönherr, Asja Fischer, Dorothea Kolossa,
> Thorsten Holz. **ICML 2020** (*International Conference on Machine Learning* — a
> top machine-learning conference). arXiv:2003.08685.

It first showed that GAN-generated images carry a grid-like artefact in the DCT
spectrum. Our custom DCT layer is a "learnable" version of their fixed analysis.

Supporting papers that back our conclusion are listed in `docs/related-work.md`
(SBI, CVPR 2022; Ojha et al., CVPR 2023; Gragnaniello et al., ICME 2021; Corvi et
al., ICASSP 2023).

## 6. Where the results live

Everything is committed to the repo under `results/` (permanent, version-controlled):
`results/README.md` and `results/analysis/README.md` have the full tables.
Local backups: `~/Downloads/deeptrace_results_20260904.tar.gz` and
`~/Downloads/analysis_final_0117.tar.gz`.

## 7. What's left before writing the paper

1. **The c40 experiment** — train *and* test at heavy compression (F3-Net's home
   turf). ~5 hours of Kaggle GPU when the quota resets. Plan in
   `docs/phase3-plan.md`.
2. **Cross-dataset error bars** — repeat the "train here, test on a different
   deepfake dataset" check for 2 more random seeds (~35 min).
3. **Write-up** — the numbers are essentially in; draft against
   `docs/phase3-plan.md` and `docs/related-work.md`.
