# RCAC: Risk-Compliant Adaptive Conformal Calibration for Financial TSFMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](requirements.txt)
[![Reproducible](https://img.shields.io/badge/reproducibility-two%20independent%20re--runs%20identical-brightgreen)]()

Official code and data for the paper:

> **Calibrating Time-Series Foundation Models for Financial Risk:
> A Risk-Compliant Study of Post-hoc Conformal Calibration**
---

## TL;DR

Time-series foundation models (Kronos, Chronos) emit **severely under-dispersed**
prediction intervals on financial candlestick data: raw 80% intervals cover only
**42–59%** of realizations in our Chinese A-share sample (**0/9** one-sided Kupiec
passes), a property of the predictive distribution confirmed at 256 sample paths.
RCAC repairs this with a **training-free, model-agnostic, zero-tuning** post-hoc
layer built on one invariant — *feedback–modulation decoupling*: because the adaptive
level updates from misses of the *unmodulated* interval, any widen-only modulation
preserves the ACI long-run coverage guarantee and its convergence rate at zero cost.

| | Raw TSFM | **RCAC (ours)** |
|---|---|---|
| One-sided Kupiec passes (9 cells, 80/90/95%) | 0/9 | **6/9** (plain ACI: 3/9) |
| Tail depth-to-width ratio, CSI 300 / 500 / Moutai | 0.49 / 0.50 / 1.36 | **0.08 / 0.09 / 0.17** |
| Exception-penalty rate (per 100 steps), CSI 300 | — | **2.4** (plain ACI: 10.2) |
| Training / validation required | — | **none** (fixed γ=0.10, cap=2.0) |

We report the boundaries as plainly as the wins: retrained references (EnbPI,
GARCH-t) win width-adjusted scores; no method passes first-order conditional
coverage; at the 80% level alone RCAC is compliance-equivalent to plain ACI.

## One-command reproduction

```bash
git clone https://github.com/hanxiaole666/RCAC-TSFMs.git
cd RCAC-TSFMs
pip install -r requirements.txt

# fetch the Kronos model package (code only; weights auto-download on first run)
git clone https://github.com/shiyu-coder/Kronos.git
xcopy /E /I Kronos\model model      # Windows; cp -r Kronos/model model on Linux

python run_all.py                   # ~60-70 min on RTX 2060 6GB, resumes on interruption
```

`run_all.py` executes 15 pipeline stages with checkpointing (existing outputs are
skipped; `--force` reruns everything; `--stage N` resumes from stage N).
Every table/figure of the paper maps to a stage — see the header of `run_all.py`.
**All scripts default to `PROJECT_ROOT = D:\finrisk_project`** (one variable at the
top of each script; change it to any path you like). Model weights (Kronos-mini,
Chronos-small) and market data auto-download on first use and are cached locally.

### Environment (verified)

| | |
|---|---|
| OS | Windows 10 / Anaconda |
| Python | 3.11.5 |
| PyTorch | 2.2.1+cu121 |
| GPU | single RTX 2060, 6 GB |
| Total pipeline | < 70 min (CPU also works, ~6× slower) |

Two independent full runs produced **bit-identical tables** (fixed seeds 42/43/44;
deterministic data caching). Total artifact footprint < 1 GB.

## Repository layout

```
RCAC-TSFMs/
├── run_all.py                  # one-command master pipeline (checkpointed)
├── requirements.txt
├── reproduction/
│   ├── 00_local_ai_bench.py    # hardware sanity check
│   ├── 01_kronos_risk_test.py  # v1: GARCH-synthetic sanity + first under-dispersion signal
│   ├── 02_kronos_risk_test_v2.py # real CSI 300 + quantile intervals + conformal fix
│   ├── 03_kronos_risk_test_v3.py # ACI adaptive calibration (the turning point)
│   ├── 04_kronos_risk_test_v4.py # 2x2 ablation: SYMMETRIC modulation NEGATIVE RESULT
│   ├── 05_kronos_risk_test_v5.py # asymmetric modulation + Chronos + gamma sensitivity
│   ├── 06_kronos_risk_test_v6.py # MAIN: 11 methods x 3 assets x 3 levels, nested protocol
│   ├── 07_dm_test.py           # Diebold-Mariano with block bootstrap (Table 4)
│   ├── 08_fix_and_summarize_v6.py # CI orientation fix + pooled summaries
│   ├── 09_conditional_diagnostics.py # volatility-tercile conditional coverage (Table B2)
│   ├── 10_es_backtest.py       # tail depth-to-width ES diagnostic (Table B4)
│   ├── 11_window_sensitivity.py# lookback/horizon robustness (Table B5)
│   ├── 12_path_sens_256.py     # 16/64/256-path confirmation
│   ├── 13_make_figures.py      # Fig 1-4
│   └── 14_make_table5.py       # per-asset RCAC aggregation helper
├── results/
│   └── v6_onesided_kupiec.csv  # 99-row authoritative per-cell table (shipped;
│                               # every compliance count in the paper derives from it)
├── docs/
│   ├── paper_manuscript.docx
│   ├── contamination_audit.md  # pretraining-cutoff audit checklist + author email template
│   └── responses/              # full response-to-reviewers history (R2-R5)
├── data/                       # (empty; scripts fetch via akshare and cache here)
└── results/                    # (populated by run_all.py)
```

## Pipeline → paper mapping

| Stage | Script | Paper output |
|---|---|---|
| 01 | `01_kronos_risk_test.py` | Sec. 5.1 sanity check (Fig. 2/3 drafts) |
| 02–05 | v2–v5 | Ablations O2/O3; Table B2; negative results |
| **06** | `06_kronos_risk_test_v6.py` | **Tables 2a, 2b, B3** (main results) |
| 07 | `07_dm_test.py` | **Table 4** (block-bootstrap DM) |
| 08 | `08_fix_and_summarize_v6.py` | corrected CIs, pooled summaries |
| 09 | `09_conditional_diagnostics.py` | **Table B2** |
| 10 | `10_es_backtest.py` | **Table B4** |
| 11 | `11_window_sensitivity.py` | **Table B5** |
| 12 | `12_path_sens_256.py` | Sec. 5.9 (path robustness) |
| 13 | `13_make_figures.py` | **Figures 1–4** |

## Key design decisions (why the code looks the way it does)

- **Nested protocol**: windows 1–5 validate (select γ, cap), windows 6–10 test.
  ACI state and residual pools evolve continuously from validation into test,
  simulating a deployment that has already run five months. No test information
  leaks into any design choice.
- **Seeds are NOT observations**: a cell = (asset, level) with n = 100 market steps.
  The 3 sampling seeds regenerate forecasts on the *same* realized path; counting
  them toward backtest sample size would be pseudo-replication. This is why every
  binomial test uses n = 100.
- **Per-asset width metrics**: price-unit quantities (Winkler, exceedance, ES
  ratio) are reported per asset, never pooled across the ~5× price-level gap
  (CSI 500 ~6000 pts vs Moutai ~1500 CNY). Pooled width numbers in earlier
  drafts were an artifact; this repo's tables are the corrected per-asset versions.
- **The 99-row CSV is the single source of truth** for every compliance count
  (abstract, Tables 2a/2b, Sec. 5.2/5.3, conclusion). We verified string-level
  consistency between the manuscript and this file.

## Data & model provenance

- **Market data**: `akshare` (open-source interface to Chinese public market data).
  Cached CSVs carry SHA-256 entries in `results/manifest` (generated by stage 06).
- **TSFMs**: Kronos-mini (4.1M params, `NeoQuasar/Kronos-mini`) and Chronos-small
  (`amazon/chronos-t5-small`) via Hugging Face. Pretraining-cutoff audit:
  `docs/contamination_audit.md`.
- **Baselines**: GARCH-t (`arch`), CQR & EnbPI (`scikit-learn`), historical
  simulation — all reimplemented from their references in `06_`.

## Honest limitations (mirrors paper Sec. 6.2)

- One market (China A-share), daily frequency, three assets. Cross-market claims
  are **not** made.
- n ≈ 100 steps/cell: one-sided Kupiec resolves only ≈10 pp deviations at 80%.
  A "pass" = absence of *detectable* under-coverage, never certified compliance.
- First-order conditional coverage fails for **every** calibration-layer method
  (0/9); this is reported as a class-level open problem.
- Formal McNeil–Frey ES testing and single-tailed 95/99% VaR backtests are beyond
  the central-interval design (the Table B4 metric is a violation-depth diagnostic).
- Chronos validation is single-seed at the 80% level (preliminary).

## Citation

```bibtex
@article{rcac2026,
  title   = {Calibrating Time-Series Foundation Models for Financial Risk:
             A Risk-Compliant Study of Post-hoc Conformal Calibration},
  author  = {Anonymous},
  journal = {submitted to IEEE TKDE},
  year    = {2026},
  note    = {Code: https://github.com/hanxiaole666/RCAC-TSFMs}
}
```

## License

MIT for code (this repository). Market data via akshare under its terms of use;
model weights under their respective Hugging Face licenses. Not investment advice.
