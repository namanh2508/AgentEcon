# Experiment Guide — AgentEcon (post-revision, USD 300 / 8 weeks)

This guide reproduces every empirical result reported in `paper.tex` (Figures 3--7, Tables I--III). Total cost is capped at **USD 300** with a calendar of **8 weeks** wall-clock, on a single workstation plus a small AWS Fabric WAN testbed.

---

## 1. Hardware and software

### 1.1 LAN testbed (lab-owned, USD 0)
- Workstation (có thể hỗ trợ các em chạy GPU A100 của UIT)
- Disk: 2 TB NVMe.

### 1.2 Fabric WAN testbed (AWS, paid)
- 4 × `c6i.xlarge` validator peers in `us-east-1` (single region, for cost discipline).
- 1 × `t3.medium` orderer.
- 1 × S3 bucket for ledger snapshots (50 GB working set).

### 1.3 Software
- Hyperledger Fabric 2.5.6 (chaincode in Go 1.22).
- Python 3.11.9, vLLM 0.6.0, PyTorch 2.4.0, NumPy 1.26, SciPy 1.13, pandas 2.2, statsmodels 0.14.
- LLMs (open weights, Hugging Face): `meta-llama/Meta-Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, `Qwen/Qwen2.5-7B-Instruct`.
- Reproducibility: deterministic seeds 1234, 2345, 3456, 4567, 5678; CUDA non-deterministic kernels disabled in PyTorch.

---

## 2. Datasets

| Asset | Source | Cadence | Sample size |
|---|---|---|---|
| Gold (XAU/USD) | Yahoo Finance public, 2018–2025 | Daily | 1820 obs. |
| WTI Oil | EIA public, 2018–2025 | Daily | 1820 obs. |
| EUR/USD FX | ECB statistical, 2018–2025 | Daily | 1820 obs. |
| Real-estate index | Case–Shiller (FRED), 2018–2025 | Daily-resampled | 1820 obs. |

All four are real public time series; **ground truth is the actual published settlement**, never a reference oracle (R4 circular-methodology concern addressed). Each settlement query is constructed from the prior 7-day window of OHLC + a structured news context.

Action grid: $\mathcal{X}$ = 1024 uniform bins per asset over the empirical 7-year price domain, 10-bit precision (R4 action-space-discretisation concern addressed).

---

## 3. Reproduction plan (8 calendar weeks, USD 300 hard cap)

### Week 1 — Environment setup, data acquisition, calibration
- Install Fabric 2.5.6 on LAN testbed; deploy AgentEcon chaincode.
- Pull the four datasets; structure into prompt template.
- Calibrate `Stylised Logit-QRE samplers` against actual LLM inference on a 200-query held-out subset for each of the three LLMs (measure $\eta_0$). The stylised sampler is then used for cheap configurations; $\eta_0 \le 0.04$ TV in all our calibrations is required for a configuration to be reported as `Stylised QRE`. Every figure in the paper distinguishes `LLM` and `Stylised QRE` in its legend (R4 conflation concern addressed).

### Week 2 — Honest-reporting fraction sweep (Fig. 3)
- 5 seeds × 5 $\tau$ × 5 $\lambda$ × 4 tasks × 1 LLM (Llama-3.1-8B) = 500 cells.
- 1 000 settlement queries per cell; 4 actual-LLM validators per cell, 60 stylised per cell.
- Total Llama-3.1 inference: 500 × 1 000 × 4 = 2 000 000 calls; ≈ 10 GPU-hours on A6000 (310 ms/call).

### Week 3 — Multi-LLM consistency (Fig. 5)
- Replicate the Week-2 protocol with Mistral-7B-Instruct-v0.3 and Qwen2.5-7B-Instruct on a smaller grid (3 $\tau$ × 3 $\lambda$ × 4 tasks × 5 seeds × 2 LLMs = 360 cells).
- Total inference for the two extra LLMs: 360 × 1 000 × 4 × 2 ≈ 2 880 000 calls ≈ 12 GPU-hours combined.

### Week 4 — Mixed-population / welfare-gap scaling (Fig. 4)
- 5 seeds × 5 $\lambda$ × 4 tasks at $\tau$ = 0.7, 1 LLM (Llama-3.1-8B). 100 cells × 1 000 × 4 = 400 000 calls ≈ 2 GPU-hours.

### Week 5 — Prompt-injection robustness (Fig. 6)
- Build TV-bounded jailbreak corpus using the multi-turn variant of Liu et al.'s benchmark (no synthesis, only filtering for TV bands).
- 5 seeds × 5 $\eta_{\mathrm{adv}}$ × 4 tasks at $\tau$ = 0.7, $\lambda$ = 0.5, 1 LLM. 100 cells × 1 000 × 4 = 400 000 calls ≈ 2 GPU-hours.

### Week 6 — Sybil / reputation dynamics (Fig. 7) and ablation (Table III)
- 3 $n_S$ × 5 seeds × 120 rounds (Sybil decay).
- Ablation: 5 configurations × 5 seeds × 1 000 queries.
- Total: ≈ 6 GPU-hours.

### Week 7 — Fabric WAN deployment, chaincode benchmarking
- Deploy chaincode on AWS WAN; measure end-to-end commit latency at 64-validator scale.
- 60 hours of `c6i.xlarge` × 4 instances = 240 instance-hours on AWS.
- Cost: 240 × USD 0.170 = **USD 40.80**.
- Orderer: 60 hours × USD 0.0416 = **USD 2.50**.
- S3 storage 50 GB × 2 months × USD 0.023 = **USD 2.30**.
- Egress 100 GB × USD 0.09 = **USD 9.00**.
- CloudWatch / IAM = **USD 4.00**.

### Week 8 — Statistical analysis and figure generation
- Paired-bootstrap 95 % CI (10 000 resamples) on every reported number.
- Holm–Bonferroni paired t-test across 9 baseline comparisons at $\alpha = 0.05/9 \approx 5.6\times 10^{-3}$.
- Two-way ANOVA on model × method interaction (Fig. 5).
- Generate all `paper.tex` figures via `tikz` and `pgfplots` using deterministic `csv` data tables exported to `data/`.

---

## 4. Total compute and cost summary

| Item | Hours | Cost (USD) |
|---|---:|---:|
| LAN A6000 GPU (lab-owned) | 32 hours | 0.00 |
| AWS `c6i.xlarge` × 4 (WAN testbed) | 240 instance-h | 40.80 |
| AWS `t3.medium` orderer | 60 h | 2.50 |
| AWS S3 (50 GB × 2 months) | — | 2.30 |
| AWS egress (100 GB) | — | 9.00 |
| AWS CloudWatch/IAM | — | 4.00 |
| Hugging Face Pro (model pull) 2 mo | — | 18.00 |
| Weights & Biases (free tier) | — | 0.00 |
| Buffer for re-runs and statistical-rigor follow-ups | — | 223.40 |
| **Total** | | **300.00** |
| **Calendar** | | **8 weeks** |

The buffer absorbs (a) rerun of the multi-LLM consistency study at higher granularity, (b) one additional asset task if reviewers request, (c) AWS spot-pricing variation up to 50 %.

---

## 5. Statistical-rigor protocol (R4 concerns)

1. **Five seeds**: 1234, 2345, 3456, 4567, 5678. Each table/figure value reports mean over five seeds.
2. **Confidence intervals**: 95 % paired-bootstrap with 10 000 resamples.
3. **Hypothesis tests**: Holm–Bonferroni-corrected paired $t$-test across 9 baseline comparisons; significance level $\alpha = 5.6 \times 10^{-3}$.
4. **Reporting**: every figure shows error bars; every tabular number reports $\pm$ CI half-width.
5. **Effect-size**: Cohen's $d$ reported for all main-effect comparisons (target $d \ge 0.5$).

---

## 6. Reproducibility artefacts

- Code repository: `https://github.com/anonymous/agentecon` (submitted as supplementary; anonymised for review).
- Data: all four time-series CSVs included; preprocessing scripts deterministic.
- Trained logs: 5-seed run logs in `runs/` with deterministic hashes.
- Chaincode: `chaincode/agentecon/` (Go); deployment scripts in `fabric-deploy/`.
- Fabric WAN scripts: `aws-deploy/` (Terraform).

The repository should reproduce all figures within ±2 % numerical tolerance in 32 GPU-hours.

---

## 7. Scoping discipline

The protocol deliberately excludes:
- Sub-second latency benchmarks (out of settlement-tier scope; see Section IV-C of the paper).
- zkML-attested LLM inference (complementary; see Section VII discussion).
- Multi-region WAN deployments beyond `us-east-1` (cost reason).
- Closed-weights LLMs (reproducibility concern).

These are documented in Section VII of the paper as future work.
