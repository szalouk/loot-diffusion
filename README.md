# Empowering Masked Diffusion Models to Self-Correct with Leave-One-Out Transformers

**Sofian Zalouk, Vincent Counathe, Paul Jünger, Daniel Cao, Daniel Zlotnick, Kilian Q. Weinberger, Christopher De Sa**

Department of Computer Science, Cornell University

**Accepted at NeurIPS 2026.**

[![OpenReview](https://img.shields.io/badge/OpenReview-ic1GS1Kl0Z-8c1515)](https://openreview.net/forum?id=ic1GS1Kl0Z)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](https://openreview.net/pdf?id=ic1GS1Kl0Z)
[![Code](https://img.shields.io/badge/Code-coming%20soon-lightgrey)](#code-release)

> **Code, checkpoints, and sampling scripts will be released in this repository soon.**

## TL;DR

A hollow transformer architecture and training objective that let masked diffusion language models train on every token and self-correct at inference, with a single network acting as both predictor and corrector.

## Abstract

Masked diffusion models (MDMs) have emerged as a promising paradigm for language modeling, enabling parallel token generation with competitive performance. Yet standard MDMs suffer from two fundamental limitations: (1) inability to revise tokens, and (2) sparse training signal. We revisit hollow transformers, which address both limitations by predicting each token from its surrounding context, independently of the input at that position. We introduce the Leave-One-Out Transformer (LOOT), a more efficient and expressive architecture that advances hollow transformers through two complementary advances. First, we redesign the merging mechanism to enable frequent interaction between left and right contexts, substantially improving predictive performance over prior hollow transformers. Second, we derive a lower-variance estimator of the MDM NELBO that supervises all tokens and improves validation perplexity. Together, these advances yield a competitive standalone diffusion language model capable of effective self-correction during inference. On LM1B, LOOT matches the strongest dense-supervision MDM baseline in validation perplexity. On OpenWebText, LOOT achieves higher MAUVE and better generation quality-diversity tradeoffs than prior remasking methods, using 8× fewer forward evaluations and 3.5× fewer inference FLOPs.

## Method

### The leave-one-out property

A standard bidirectional denoiser can see the token it is asked to predict, so visible tokens can never be supervised and never be revised. LOOT's output at position *i* depends on every input token except *x<sub>i</sub>*, which makes every position a valid training target and gives the model a conditional distribution over replacements for every token, masked or not.

<p align="center">
  <img src="assets/dependency_patterns.png" width="85%" alt="Dependency patterns of causal, bidirectional, and leave-one-out models">
</p>

### Architecture

LOOT keeps three streams of hidden states: a left-to-right stream, a right-to-left stream, and an output stream that merges the two at **every layer** while never attending to itself, so the leave-one-out property holds under composition. All three streams share weights, and the merge is a sliding-window attention that runs on unmodified FlashAttention kernels, which is what makes merging at every layer affordable. The parameter count matches a standard diffusion transformer of the same depth and width, and a pretrained MDM checkpoint can initialize all three streams.

<p align="center">
  <img src="assets/loot_architecture.png" width="95%" alt="LOOT architecture: forward, backward, and output streams">
</p>

### Training objective

Because the prediction at position *i* does not depend on whether *x<sub>i</sub>* is masked, the per-position masking probability can be marginalized analytically inside the masked-diffusion ELBO. The resulting objective supervises every token with the bounded weight *&alpha;'<sub>t</sub>*, equals the MDLM NELBO in expectation, and has strictly lower variance than both the MDLM objective and the two-term objectives used to train hollow transformers previously.

### Self-correction

One forward pass returns the leave-one-out conditional at every position, which is exactly what classical conditional samplers need. LOOT revises its own samples with blocked Gibbs sampling or a Metropolis-style corrector, with no remasking heuristics and no separate corrector model.

<p align="center">
  <img src="assets/repair_illustration.png" width="60%" alt="Self-correction in MDM, ReMDM, and LOOT">
</p>

## Headline results

### Language modeling (validation perplexity, lower is better)

Parameter-matched comparison; LM1B at sequence length 128 for 1M steps, OpenWebText at length 1024 for 250K steps. Mean &plusmn; std over 5 evaluation seeds.

| Method | LM1B | OpenWebText |
|---|---|---|
| MDLM | 31.80 &plusmn; 0.08 | **25.20 &plusmn; 0.02** |
| PGM | 31.04 &plusmn; 0.09 | 26.57 &plusmn; 0.02 |
| **LOOT** | **31.02 &plusmn; 0.04** | 26.83 &plusmn; 0.01 |

On LM1B, LOOT matches PGM and improves over MDLM. At sequence length 1024 a gap to MDLM remains: the cost of the leave-one-out constraint grows with length while the benefit of dense supervision stays roughly constant.

### Self-correction on OpenWebText

Self-correction is where the architecture pays off. On unconditional generation at length 1024, LOOT with self-correction at **256 NFE** matches or exceeds ReMDM at **2048 NFE** across the full quality-diversity frontier, using **8x fewer forward evaluations** and, after accounting for the 2.2x cost of a LOOT forward pass, **3.5x fewer inference FLOPs** and about **8x less wall-clock time per sample**.

<p align="center">
  <img src="assets/frontier_loot_vs_remdm.png" width="62%" alt="LOOT at 256 and 512 NFE versus ReMDM at 2048 NFE">
  <img src="assets/mauve_vs_nfe.png" width="34%" alt="MAUVE across NFE budgets">
</p>

Left: generative perplexity against sample entropy over a temperature sweep. Right: MAUVE on 5000 samples per point, under the evaluation protocol of ReMDM. At the entropy of natural text, blocked Gibbs at 256 NFE reaches a generative perplexity of 23.7 against 41.3 for ReMDM at 2048 NFE, and a MAUVE of 0.41 against 0.35.

<p align="center">
  <img src="assets/frontiers_all_nfe.png" width="95%" alt="Quality-diversity frontiers at every NFE budget">
</p>

Across every budget from 64 to 2048 NFE, LOOT self-correction shifts the frontier outward relative to MDLM, ReMDM, and the uncorrected LOOT sampler.

## Code release

Code, pretrained checkpoints, and the sampling and evaluation scripts are being prepared for release in this repository. Watch or star the repository to be notified.

## Citation

```bibtex
@inproceedings{zalouk2026loot,
  title     = {Empowering Masked Diffusion Models to Self-Correct with Leave-One-Out Transformers},
  author    = {Zalouk, Sofian and Counathe, Vincent and J{\"u}nger, Paul and Cao, Daniel and Zlotnick, Daniel and Weinberger, Kilian Q. and De Sa, Christopher},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Contact

Sofian Zalouk, saz43@cornell.edu
