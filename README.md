# AgentEcon: LLM-Based Economic Agent Prediction Market Oracle

AgentEcon là repo thực nghiệm để tái lập ý tưởng trong `AgentEcon_G02.pdf`: dùng LLM như các tác nhân kinh tế trong prediction-market oracle, kết hợp Stylised Logit-QRE, quadratic scoring, reputation, và Hyperledger Fabric.

Repo này hiện hỗ trợ các phần nền tảng: data pipeline, structured GDELT news cache, QRE/stylised simulation, vLLM predictor, mixed runner `4 LLM + 60 stylised`, LLM/QRE calibration script, prompt-injection runner, Sybil/reputation runner, ablation runner, oracle simulator, Go chaincode build, thống kê, figure generation, Docker/Terraform skeleton. Các phần còn phụ thuộc hạ tầng ngoài là vLLM/model weights cho actual LLM inference và Fabric/AWS thật cho blockchain latency.

## Yêu Cầu Hệ Thống

### Máy local / WSL

- WSL2 Ubuntu 22.04 hoặc Linux native.
- Python 3.10 hoặc 3.11. Guide gốc ghi Python 3.11.9; repo đã được kiểm tra với Python 3.10 trong WSL.
- Go 1.22+ để build chaincode.
- Docker + Docker Compose để kiểm tra Fabric compose.
- Terraform nếu chạy phần AWS.

### Máy GPU cho vLLM

- Stylised/QRE không cần GPU.
- vLLM 7B/8B cần NVIDIA CUDA GPU mạnh.
- Khuyến nghị A100/A6000 cho protocol lớn trong guide.
- RTX 3090/4090 24GB có thể chạy pilot nhỏ với batch nhỏ và `max_model_len` giảm.
- Laptop GPU 6GB thường không đủ cho Llama/Mistral/Qwen 7B-8B bằng vLLM FP16.
- Disk nên có 250GB+ nếu tải cả 3 Hugging Face model và cache.

## Cấu Trúc Repo

```text
AgentEcon/
├── src/
│   ├── data/asset_loader.py        # Tải/chuẩn hóa 4 asset, tạo settlement queries
│   ├── data/news_loader.py         # Fetch/cache GDELT news context, hash raw/processed context
│   ├── models/predictors.py        # Prompt template, parser, vLLM predictor, Logit-QRE sampler
│   ├── oracle/fabric_oracle.py     # Oracle simulator: submit, settle, scoring, reputation, sybil trimming
│   ├── experiments/runner.py       # ExperimentConfig + ExperimentRunner
│   └── analysis/statistics.py      # Bootstrap CI, Holm-Bonferroni, Cohen's d, ANOVA
├── scripts/
│   ├── download_data.py            # Tải data và sinh query JSON
│   ├── run_llm_calibration.py      # Chạy Week-1 LLM/QRE calibration, xuất eta0/TV/latency
│   ├── download_news_contexts.py   # Prefetch structured news context
│   ├── run_prompt_injection.py     # Sinh dữ liệu Figure 6
│   ├── run_sybil_reputation.py     # Sinh dữ liệu Figure 7
│   ├── run_ablation.py             # Sinh dữ liệu Table III
│   ├── analyze_results.py          # Tạo statistical_report.json từ CSV kết quả
│   ├── generate_figures.py         # Tạo Figure 3-7 dạng PNG/PDF từ CSV kết quả
│   └── smoke_test.py               # Smoke test Python trực tiếp
├── config/experiment_config.yaml   # Config mặc định theo guide: assets, seeds, tau, lambda, models
├── chaincode/                      # Go chaincode cho Fabric
├── fabric-deploy/                  # Docker Compose + script deploy chaincode
├── aws-deploy/                     # Terraform AWS WAN testbed
├── data/                           # raw, processed, runs
├── figures/                        # output hình
├── requirements.txt                # Python stack nhẹ
└── requirements-llm.txt            # Stack vLLM/PyTorch CUDA nặng
```

## Quick Start: Chạy Không Cần GPU

Chạy trong WSL/Linux từ thư mục repo.

```bash
cd /path/to/AgentEcon

make install
make setup-data
make test
make run-smoke
make run-pilot
```

Các lệnh trên dùng môi trường `.venv-wsl` trên Linux/WSL và chạy `mode="stylised"`, phù hợp để kiểm tra code, data pipeline, QRE sampler, oracle simulator và logging mà không cần GPU.

Output chính:

- `data/raw/`: CSV raw/local cache.
- `data/processed/*_queries.json`: settlement queries đã sinh.
- `data/runs/*_summary.json`: summary mỗi lần chạy.
- `data/runs/*.log`: log experiment.

## Cài Đặt Chi Tiết

### 1. Python environment nhẹ

```bash
cd /path/to/AgentEcon
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Kiểm tra:

```bash
python -m pytest tests -q
```

### 2. Data pipeline

Tải/chuẩn hóa 4 asset và sinh query:

```bash
source .venv-wsl/bin/activate
python scripts/download_data.py --n-queries 100
```

Tham số `--n-queries` là số query mỗi asset mỗi seed. Guide full run dùng 1,000 settlement queries/cell; để chuẩn bị query ở quy mô lớn:

```bash
python scripts/download_data.py --n-queries 1000
```

Asset theo `experiment_guide.md`:

| Asset key | Ý nghĩa | Source theo loader/guide |
|---|---|---|
| `XAU_USD` | Gold | Yahoo/public CSV fallback |
| `WTI` | Oil | EIA/public CSV fallback |
| `EUR_USD` | FX | ECB/Yahoo/public CSV fallback |
| `CASE_SHILLER` | Real-estate index | FRED/local fallback, daily-resampled |

Mỗi query có prior 7-day OHLC, settlement date, ground-truth settlement, `bin_id` trên grid 1024, `price_domain`, `news_context`, `news_context_hash`, `news_source`, `news_window`, và `n_news_items`. Mặc định `news_enabled=False` để chạy nhanh/offline; muốn dùng structured news thật thì chạy phần news bên dưới.

### 2b. Structured news context

News context dùng GDELT DOC 2.1, cache raw response trong `data/raw/news/` và processed context trong `data/processed/news_contexts/`. Chạy prefetch:

```bash
source .venv-wsl/bin/activate
python scripts/download_news_contexts.py \
  --assets XAU_USD WTI EUR_USD CASE_SHILLER \
  --seeds 1234 \
  --n-queries 10 \
  --max-items 5
```

Output:

```text
data/raw/news/*_gdelt.json
data/processed/news_contexts/*.json
data/processed/news_contexts_manifest.json
```

GDELT có thể rate-limit hoặc timeout. Khi fetch lỗi, raw cache vẫn ghi `fetch_error` và processed context có `n_news_items=0`; các context này giúp pipeline không vỡ nhưng không nên dùng để claim là có news evidence. Trước khi báo cáo structured-news experiment, kiểm tra `data/processed/news_contexts_manifest.json` và chỉ dùng các query có `n_news_items > 0`.

Để sinh query có news thật trong code:

```python
from src.data.asset_loader import AssetLoader

loader = AssetLoader(data_dir="data", news_enabled=True)
queries = loader.generate_queries("XAU_USD", n_queries=100, seed=1234)
```

### 3. Go chaincode và Fabric config

Build chaincode:

```bash
make build-chaincode
```

Kiểm tra Docker Compose config:

```bash
docker compose -f fabric-deploy/docker-compose.yaml config
```

Chạy Fabric local skeleton:

```bash
make fabric-start
make fabric-logs
make fabric-stop
```

Lưu ý: phần Python hiện dùng `FabricOracleClient` dạng simulator trong `src/oracle/fabric_oracle.py`. Để chạy Fabric thật cần hoàn thiện network artifacts/crypto/deploy lifecycle theo môi trường Fabric của bạn.

### 4. Terraform AWS WAN

Kiểm tra Terraform:

```bash
cd aws-deploy
terraform init -backend=false
terraform fmt -check
terraform validate
```

Deploy thật chỉ chạy khi đã cấu hình AWS credentials và budget alarm:

```bash
terraform plan
terraform apply
```

Theo guide, AWS WAN dùng 4 physical `c6i.xlarge` peer + 1 `t3.medium` orderer. Benchmark "64-validator scale" trong repo nên được hiểu là 64 logical validator identities submit qua 4 peers, mặc định 16 logical validators/peer, không phải 64 EC2 validator vật lý.

## Cài vLLM Trên Máy GPU Mạnh

Không cài vLLM vào Python gốc. Luôn dùng venv của repo.

```bash
cd /path/to/AgentEcon
source .venv-wsl/bin/activate

python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements-llm.txt
```

`requirements-llm.txt`:

```txt
torch==2.4.0
vllm==0.6.0
transformers==4.44.2
```

Không nâng `transformers` tự do lên 5.x với `vllm==0.6.0`, vì có thể lỗi import với `torch==2.4.0`.

Verify:

```bash
python - <<'PY'
import torch
import transformers
import vllm

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
PY

python -m pip check
```

### Tải 3 model Hugging Face

Đăng nhập Hugging Face trước. Llama có thể cần request access trên Hugging Face.

```bash
source .venv-wsl/bin/activate
python -m pip install "huggingface-hub[cli]==0.36.2"
huggingface-cli login

export HF_HOME=/mnt/data/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub

huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

Đặt `HF_HOME` ở ổ nhiều dung lượng, không đặt model cache trong repo.

## Chạy Code: File Nào Làm Gì, Lệnh Nào Dùng

| Mục tiêu | File chính | Lệnh |
|---|---|---|
| Cài môi trường nhẹ | `requirements.txt`, `Makefile` | `make install` |
| Tải data + sinh query | `scripts/download_data.py`, `src/data/asset_loader.py` | `python scripts/download_data.py --n-queries 100` |
| Tải/cache structured news | `scripts/download_news_contexts.py`, `src/data/news_loader.py` | `make setup-news` |
| Unit tests | `tests/test_agentecon.py` | `make test` hoặc `python -m pytest tests -q` |
| Smoke test stylised | `src/experiments/runner.py` | `make run-smoke` |
| Pilot stylised 200 query | `src/experiments/runner.py` | `make run-pilot` |
| LLM/QRE calibration 200 query/model | `scripts/run_llm_calibration.py` | `make run-calibration` hoặc lệnh chi tiết bên dưới |
| Full stylised grid qua Makefile | `src/experiments/runner.py` | `make run-experiment` |
| Mixed mini `4 LLM + 60 stylised` | `src/experiments/runner.py` | `make run-mixed-mini` |
| LLM predictor | `src/models/predictors.py` | dùng Python snippet ở dưới |
| Statistical report | `scripts/analyze_results.py`, `src/analysis/statistics.py` | `python scripts/analyze_results.py --data-dir data/runs --results-file <file>.csv` |
| Figures | `scripts/generate_figures.py` | `python scripts/generate_figures.py --data-dir data/runs --results-file <file>.csv --output-dir figures/<run>` |
| Figure 6 prompt injection | `scripts/run_prompt_injection.py` | `make run-prompt-injection` |
| Figure 7 Sybil/reputation | `scripts/run_sybil_reputation.py` | `make run-sybil` |
| Table III ablation | `scripts/run_ablation.py` | `make run-ablation` |
| Build chaincode | `chaincode/agentecon.go` | `make build-chaincode` |
| Fabric local skeleton | `fabric-deploy/docker-compose.yaml` | `make fabric-start` |
| AWS WAN skeleton | `aws-deploy/main.tf` | `make aws-init`, `make aws-plan`, `make aws-deploy` |

## Chạy LLM Pilot Thật Bằng vLLM

Chạy thử 1 query trước:

```bash
source .venv-wsl/bin/activate

python - <<'PY'
import pandas as pd

from src.data.asset_loader import AssetLoader
from src.models.predictors import LLMPredictor

asset = "XAU_USD"
model = "meta-llama/Meta-Llama-3.1-8B-Instruct"

loader = AssetLoader(data_dir="data")
query = loader.generate_queries(asset, n_queries=1, seed=1234)[0]

predictor = LLMPredictor(
    model_name=model,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=2048,
)

result = predictor.predict(
    query_id=query.query_id,
    asset_name=loader.ASSET_CONFIGS[asset]["name"],
    query_date=(query.timestamp - pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
    settlement_date=query.timestamp.strftime("%Y-%m-%d"),
    prior_7day_df=query.prior_7day_window,
    min_price=query.price_domain[0],
    max_price=query.price_domain[1],
    grid_size=1024,
    news_context=query.news_context,
)

print(result)
PY
```

Chạy pilot 200 query cho một model:

```bash
source .venv-wsl/bin/activate

python - <<'PY'
from src.experiments.runner import ExperimentConfig, ExperimentRunner

config = ExperimentConfig(
    experiment_name="llm_pilot_llama31_8b",
    mode="llm",
    models=["meta-llama/Meta-Llama-3.1-8B-Instruct"],
    assets=["XAU_USD"],
    seeds=[1234],
    n_queries_pilot=200,
    n_validators=4,
)

summary = ExperimentRunner(config).run(scale="pilot")
print(summary["status"])
print(summary["pilot"]["avg_latency_ms"])
print(summary["pilot"]["parse_failures"])
PY
```

Chạy lần lượt 3 model mục tiêu:

```bash
source .venv-wsl/bin/activate

for MODEL in \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  mistralai/Mistral-7B-Instruct-v0.3 \
  Qwen/Qwen2.5-7B-Instruct
do
  MODEL="$MODEL" python - <<'PY'
import os
from src.experiments.runner import ExperimentConfig, ExperimentRunner

model = os.environ["MODEL"]
config = ExperimentConfig(
    experiment_name=f"llm_pilot_{model.split('/')[-1]}",
    mode="llm",
    models=[model],
    assets=["XAU_USD"],
    seeds=[1234],
    n_queries_pilot=200,
    n_validators=4,
)
summary = ExperimentRunner(config).run(scale="pilot")
print(model, summary["status"])
PY
done
```

Nếu CUDA OOM: giảm `max_model_len` trong `LLMPredictor`, giảm batch size nếu dùng `predict_batch`, giảm `gpu_memory_utilization`, hoặc chuyển sang GPU nhiều VRAM hơn.

## Chạy Calibration Theo Week 1

Calibration dùng 200 query/model/asset để đo latency, parse failure, calibrated `eta0`, và TV distance giữa LLM prediction distribution với Stylised Logit-QRE. Đây là bước bắt buộc trước khi dùng stylised QRE cho các grid lớn.

Chạy đúng protocol bằng vLLM:

```bash
source .venv-wsl/bin/activate

python scripts/run_llm_calibration.py \
  --assets XAU_USD WTI EUR_USD CASE_SHILLER \
  --models meta-llama/Meta-Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3 Qwen/Qwen2.5-7B-Instruct \
  --seed 1234 \
  --n-queries 200 \
  --target-tv 0.04 \
  --output-dir data/runs/calibration
```

Output:

```text
data/runs/calibration/calibration_results.csv
data/runs/calibration/calibration_predictions.csv
data/runs/calibration/calibration_manifest.json
```

Chỉ report một cấu hình là `Stylised QRE` nếu `is_reportable_stylised_qre=true`, tức là chạy bằng vLLM thật và `tv_distance <= 0.04`. Tùy chọn `--dry-run-stylised` chỉ dùng để test pipeline không GPU; output dry-run có `protocol_valid=false` trong manifest và không được dùng làm kết quả paper.

## Chạy Full/Stylised Grid Hiện Có

Makefile `run-experiment` hiện chạy explicit `mode="stylised"` để không vô tình khởi tạo vLLM trên máy không có GPU:

```bash
make run-experiment
```

Nếu muốn tự cấu hình full stylised grid theo guide, dùng snippet sau:

```bash
source .venv-wsl/bin/activate

python - <<'PY'
from src.experiments.runner import ExperimentConfig, ExperimentRunner

config = ExperimentConfig(
    experiment_name="stylised_full_grid",
    mode="stylised",
    models=["meta-llama/Meta-Llama-3.1-8B-Instruct"],
    assets=["XAU_USD", "WTI", "EUR_USD", "CASE_SHILLER"],
    seeds=[1234, 2345, 3456, 4567, 5678],
    tau_values=[0.5, 0.7, 1.0, 1.5, 2.0],
    lambda_values=[0.0, 0.25, 0.5, 0.75, 1.0],
    n_validators=64,
    n_queries=1000,
)

summary = ExperimentRunner(config).run(scale="full")
print(summary["run_id"])
print(summary["status"])
PY
```

## Chạy Mixed Grid `4 LLM + 60 Stylised`

Runner đã hỗ trợ `mode="mixed"` đúng theo Week 2: mỗi settlement query có `n_llm_validators=4` actual LLM validators và `n_stylised_validators=60` QRE validators, tổng `n_validators=64`.

Mini mixed run:

```bash
source .venv-wsl/bin/activate
make run-mixed-mini
```

Hoặc chạy trực tiếp:

```bash
source .venv-wsl/bin/activate

python - <<'PY'
from src.experiments.runner import ExperimentConfig, ExperimentRunner

config = ExperimentConfig(
    experiment_name="mixed_llama31_xau",
    mode="mixed",
    models=["meta-llama/Meta-Llama-3.1-8B-Instruct"],
    assets=["XAU_USD"],
    seeds=[1234],
    tau_values=[0.7],
    lambda_values=[0.5],
    n_queries=1000,
    n_validators=64,
    n_llm_validators=4,
    n_stylised_validators=60,
)

summary = ExperimentRunner(config).run(scale="full")
print(summary["run_id"])
print(summary["status"])
PY
```

CSV kết quả có thêm các cột `n_llm_validators`, `n_stylised_validators`, `validator_honest_reporting_fraction`, `validator_welfare_gap`, `parse_failures`, `avg_llm_latency_ms`, và `avg_stylised_latency_ms`.

## Chạy Figure 6: Prompt-Injection Robustness

Corpus mặc định nằm ở `config/prompt_injection_attacks.jsonl`. Runner local hiện dùng stylised/QRE để kiểm tra pipeline và sinh CSV có `eta_adv`; nếu muốn đo actual LLM robustness thì cần chuyển sang LLM-backed inference sau khi đã cài vLLM/model weights.

```bash
source .venv-wsl/bin/activate
python scripts/run_prompt_injection.py \
  --assets XAU_USD WTI EUR_USD CASE_SHILLER \
  --seeds 1234 2345 3456 4567 5678 \
  --n-queries 1000 \
  --output-dir data/runs/prompt_injection
```

Output:

```text
data/runs/prompt_injection/prompt_injection_results.csv
data/runs/prompt_injection/prompt_injection_manifest.json
```

Sinh Figure 6:

```bash
python scripts/generate_figures.py \
  --data-dir data/runs/prompt_injection \
  --results-file prompt_injection_results.csv \
  --output-dir figures/prompt_injection
```

## Chạy Figure 7: Sybil/Reputation Dynamics

Runner mô phỏng `3 n_S x 5 seeds x 120 rounds`, xuất các cột `n_sybil`, `round`, `reputation_mean_honest`, `reputation_mean_sybil`, `sybil_trimmed`.

```bash
source .venv-wsl/bin/activate
python scripts/run_sybil_reputation.py \
  --asset XAU_USD \
  --seeds 1234 2345 3456 4567 5678 \
  --n-sybil-values 0 8 16 \
  --n-rounds 120 \
  --output-dir data/runs/sybil_reputation
```

Output:

```text
data/runs/sybil_reputation/sybil_reputation_results.csv
data/runs/sybil_reputation/sybil_reputation_manifest.json
```

Sinh Figure 7:

```bash
python scripts/generate_figures.py \
  --data-dir data/runs/sybil_reputation \
  --results-file sybil_reputation_results.csv \
  --output-dir figures/sybil_reputation
```

## Chạy Table III: Ablation

Runner ablation chạy 5 cấu hình: full, bỏ trimmed mean, bỏ reputation, bỏ Sybil detection, và bỏ confidence/QSR precision.

```bash
source .venv-wsl/bin/activate
python scripts/run_ablation.py \
  --asset XAU_USD \
  --seeds 1234 2345 3456 4567 5678 \
  --n-queries 1000 \
  --output-dir data/runs/ablation
```

Output:

```text
data/runs/ablation/ablation_results.csv
data/runs/ablation/ablation_manifest.json
```

Sau khi chạy xong, tìm file CSV:

```bash
ls data/runs/*_results.csv
```

Phân tích và sinh figure:

```bash
python scripts/analyze_results.py \
  --data-dir data/runs \
  --results-file <run_id>_results.csv \
  --output-dir data/runs/<run_id>_analysis \
  --n-bootstrap 10000

python scripts/generate_figures.py \
  --data-dir data/runs \
  --results-file <run_id>_results.csv \
  --output-dir figures/<run_id>
```

## Mapping Với `experiment_guide.md`

| Guide | Yêu cầu | File/lệnh trong repo | Trạng thái hiện tại |
|---|---|---|---|
| 1. Hardware/software | Python, vLLM, PyTorch, Fabric, Go, seeds | `requirements.txt`, `requirements-llm.txt`, `Makefile`, `config/experiment_config.yaml` | Có hướng dẫn setup; vLLM cần máy GPU mạnh |
| 2. Datasets | 4 public time series, 7-day OHLC, structured news, 1024 bins | `src/data/asset_loader.py`, `src/data/news_loader.py`, `scripts/download_data.py`, `scripts/download_news_contexts.py` | Có news pipeline/cache GDELT; mặc định tắt để chạy offline |
| Week 1 | Calibrate 200 query/model, đo latency/parse failure/TV | `scripts/run_llm_calibration.py`, `src/models/predictors.py` | Có script calibration; cần máy GPU/vLLM thật để output protocol-valid |
| Week 2 / Fig. 3 | 5 seeds x 5 tau x 5 lambda x 4 tasks x Llama, 1000 query/cell, 64 validators | `ExperimentRunner.run_full_grid()`, `mode="mixed"`, `scripts/generate_figures.py` | Runner đã hỗ trợ `4 LLM + 60 stylised`; cần GPU để chạy đủ grid LLM thật |
| Week 3 / Fig. 5 | Mistral + Qwen smaller grid, ANOVA model x method | `config/experiment_config.yaml`, `scripts/analyze_results.py`, `scripts/generate_figures.py` | Có multi-model field và ANOVA; cần chạy đủ CSV thực nghiệm |
| Week 4 / Fig. 4 | Welfare gap theo lambda | `src/oracle/fabric_oracle.py`, `src/experiments/runner.py`, `generate_figure4()` | Có metric và figure từ CSV |
| Week 5 / Fig. 6 | Prompt-injection robustness, jailbreak corpus | `data/raw/prompt_injection/attacks.jsonl`, `scripts/run_prompt_injection.py`, `generate_figure6()` | Có stylised runner và CSV schema; actual LLM version cần vLLM/model weights |
| Week 6 / Fig. 7 / Table III | Sybil/reputation dynamics, ablation | `scripts/run_sybil_reputation.py`, `scripts/run_ablation.py`, `generate_figure7()` | Có local simulator runners cho Fig. 7 và Table III |
| Week 7 | AWS WAN, 64-validator scale commit latency | `aws-deploy/`, `fabric-deploy/`, `avg_commit_latency_ms` | Terraform/compose skeleton có; triển khai Fabric thật cần cấu hình AWS/Fabric và chạy logical validators |
| Week 8 | Bootstrap CI, Holm-Bonferroni, Cohen's d, ANOVA, figures | `src/analysis/statistics.py`, `scripts/analyze_results.py`, `scripts/generate_figures.py` | Có pipeline phân tích; chất lượng phụ thuộc CSV đầu vào đủ cell/seeds |

## Protocol Thống Kê

- Seeds: `1234, 2345, 3456, 4567, 5678`.
- Grid: `tau=[0.5, 0.7, 1.0, 1.5, 2.0]`.
- Grid: `lambda=[0.0, 0.25, 0.5, 0.75, 1.0]`.
- Action grid: 1024 bins.
- CI: paired bootstrap 95% với 10,000 resamples.
- Hypothesis testing: Holm-Bonferroni corrected paired t-test, target alpha theo guide `0.05/9 ≈ 5.6e-3`.
- Effect size: Cohen's d, target practical effect `d >= 0.5`.
- Reproducibility tolerance: ±2% numerical tolerance.

## Output Cần Nộp / Kiểm Tra

Sau một run đủ lớn, cần có:

- `data/runs/<run_id>_results.csv`: bảng cell-level metrics.
- `data/runs/<run_id>_results.csv_manifest.json`: config, git hash, timestamp.
- `data/runs/<run_id>_summary.json`: summary run.
- `data/runs/<run_id>_analysis/statistical_report.json`: CI/test/effect size/ANOVA.
- `data/runs/calibration/calibration_results.csv`: `eta0`, TV distance, latency, parse failure cho Week 1.
- `data/runs/calibration/calibration_predictions.csv`: prediction-level log cho calibration.
- `data/processed/news_contexts_manifest.json`: manifest structured news context.
- `data/runs/prompt_injection/prompt_injection_results.csv`: dữ liệu Figure 6.
- `data/runs/sybil_reputation/sybil_reputation_results.csv`: dữ liệu Figure 7.
- `data/runs/ablation/ablation_results.csv`: dữ liệu Table III.
- `figures/<run_id>/figure3_honest_reporting.{png,pdf}`.
- `figures/<run_id>/figure4_welfare_gap.{png,pdf}`.
- `figures/<run_id>/figure5_multi_llm.{png,pdf}` nếu CSV có nhiều model.
- `figures/<run_id>/figure6_prompt_injection.{png,pdf}` chỉ có ý nghĩa khi CSV có `eta_adv`.
- `figures/<run_id>/figure7_sybil_reputation.{png,pdf}` chỉ có ý nghĩa khi CSV có `n_sybil`.

## Lộ Trình Chạy Thực Nghiệm Khuyến Nghị

1. Chạy `make install`, `make setup-data`, `make test`.
2. Chạy `make run-smoke` để kiểm tra pipeline nhanh.
3. Chạy stylised pilot `make run-pilot`.
4. Trên máy GPU mạnh, cài `requirements-llm.txt`, tải 3 model Hugging Face.
5. Chạy `scripts/run_llm_calibration.py` cho từng model và asset cần báo cáo.
6. Chỉ dùng Stylised QRE cho cấu hình có `tv_distance <= 0.04`.
7. Chạy mixed grid `4 LLM + 60 stylised` cho Fig. 3/4 baseline theo guide.
8. Chạy multi-model mixed grid nếu cần Fig. 5.
9. Chạy `scripts/run_prompt_injection.py`, `scripts/run_sybil_reputation.py`, và `scripts/run_ablation.py` cho Fig. 6/7/Table III.
10. Chỉ sau khi LAN/local pass mới chạy Fabric/AWS WAN.
11. Chạy `scripts/analyze_results.py` và `scripts/generate_figures.py` từ CSV cuối cùng.

## Các Giới Hạn Cần Ghi Rõ Khi Báo Cáo

- Structured news đã có pipeline GDELT/cache/hash, nhưng chỉ xuất hiện trong query khi chạy `AssetLoader(news_enabled=True)` hoặc prefetch bằng `scripts/download_news_contexts.py`.
- Runner đã hỗ trợ mixed population `4 actual-LLM validators + 60 stylised validators`, nhưng chỉ đúng protocol khi máy đã cài vLLM và tải model Hugging Face thật.
- Figure 6/7/Table III đã có local stylised/simulator runners. Nếu báo cáo actual LLM prompt-injection robustness thì vẫn cần vLLM/model weights.
- Fabric Python path hiện là simulator; WAN/Fabric thật cần deploy network và chaincode lifecycle đầy đủ trước khi dùng số liệu commit latency như kết quả blockchain thật.

## Tài Liệu Tham Khảo

- `AgentEcon_G02.pdf`: paper/ý tưởng gốc.
- `experiment_guide.md`: protocol 8 tuần, budget, figures/tables cần tái lập.
- `PLAN.md`: kế hoạch chuẩn bị, assumptions và các clarification đã khóa.
- McKelvey & Palfrey, 1995: Quantal Response Equilibria.
- Hyperledger Fabric 2.5 documentation.
- vLLM 0.6 documentation.
