# Justice SN56 text recipe

Seeded from the public 2026-11-16 text winner
(`gradients-opensource/god-text-tourn-e63574fc205e9b06-20261116-position-1`)
so the Docker/CLI contract stays valid. Training-time behaviour that actually
changes the loss:

1. **KL matches validator scoring.** Completion-token KL after the causal
   shift (`logits[:-1]` vs `labels[1:]`), not unshifted `labels != -100`
   indexing into raw logits. Uses `KL_COEF` when `USE_KL=1`.
2. **WSD decay 35%** of the horizon (was 20%). Shorter stable phase, longer
   anneal into the early-stop / SWA window.
3. **LR-search budget 12%** of wall time (was 20%). More steps on the real
   train set after the probe.
4. **Packed-fill LR rescale** uses `n^(0.35)` and a 1/5 member floor (was
   0.25 and 1/4).
5. **Instruct peak LR × 0.9** on every size bucket.

Do not enter image until a separate image repo exists. LICENSE.md and NOTICE
are the G.O.D originals and must stay byte-identical.
