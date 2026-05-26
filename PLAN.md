# Kế Hoạch Chuẩn Bị Dự Án AgentEcon

## Tóm Tắt

Dự án hiện có ý tưởng và guide, nhưng repo gần như chưa có code: chỉ có `AgentEcon_G02.pdf`, `experiment_guide.md`, `README.md`. Mục tiêu chuẩn bị là biến paper thành một hệ thống thực nghiệm có thể chạy lại: pipeline dữ liệu, mô phỏng LLM/QRE, chaincode Hyperledger Fabric, benchmark, thống kê, và tài liệu tái lập.

Rủi ro lớn cần khóa trước khi code: ước lượng GPU-hour trong guide có khả năng lạc quan hoặc chưa nói rõ batching; cần chạy pilot nhỏ để đo throughput thực tế trước khi cam kết full experiment.

## Các Giai Đoạn Chuẩn Bị

1. **Khóa phạm vi và baseline**
   - Xác định mục tiêu v1: tái lập tối thiểu Fig. 3 và Table III trước, sau đó mở rộng Fig. 4-7.
   - Chuẩn hóa metric: honest-reporting fraction, welfare gap, commit latency, TV shift, CI 95%.
   - Chốt seed: `1234, 2345, 3456, 4567, 5678`.
   - Tạo issue/checklist cho từng figure/table để tránh chạy thí nghiệm rời rạc.

2. **Thiết lập môi trường**
   - Dùng Linux hoặc WSL2 Ubuntu cho phần GPU/Fabric; Windows native không phù hợp cho vLLM + Fabric production-like setup.
   - Cài: Python `3.11.x`, Go `1.22`, Docker, Docker Compose, CUDA driver, NVIDIA Container Toolkit.
   - Python stack: PyTorch `2.4`, vLLM `0.6`, NumPy, pandas, SciPy, statsmodels, matplotlib/pgfplots export.
   - Blockchain stack: Hyperledger Fabric `2.5.x`, Fabric samples, Go chaincode SDK.
   - Tạo `.env.example` cho Hugging Face token, AWS credentials, EIA/FRED API keys nếu dùng API chính thức.

3. **Pipeline dữ liệu**
   - Tạo module tải và chuẩn hóa 4 asset: XAU/USD, WTI Oil, EUR/USD, Case-Shiller/FRED.
   - Lưu dữ liệu raw bất biến trong `data/raw/`, dữ liệu đã xử lý trong `data/processed/`.
   - Mỗi query phải có: prior 7-day OHLC/window, context/news field nếu có, ground truth settlement, bin ID trong grid `1024`.
   - Case-Shiller là monthly series, nên daily-resampling phải được ghi rõ phương pháp để tránh reviewer bắt lỗi “daily proxy”.

4. **Mô phỏng LLM và QRE**
   - Viết prompt template cố định, parser số deterministic, mapping output sang bin.
   - Chạy pilot 200 query/model để đo:
     - latency thực tế,
     - parse failure rate,
     - TV distance giữa LLM output và stylised Logit-QRE sampler,
     - chi phí GPU/call khi batching bằng vLLM.
   - Chỉ dùng stylised sampler cho các cell lớn sau khi TV calibration đạt ngưỡng, ví dụ `eta0 <= 0.04`.

5. **Cơ chế oracle và chaincode**
   - Implement chaincode Go với các hàm tối thiểu: `Query`, `Submit`, `Settle`, `GetReputation`.
   - Dùng fixed-point integer cho price/bin để tránh nondeterminism floating-point trong chaincode.
   - Implement trimmed weighted mean, quadratic scoring rule, reputation decay.
   - Viết unit test Go cho aggregation, scoring, reputation update, sybil trimming.

6. **Experiment runner**
   - Tạo runner cấu hình bằng YAML/JSON: model, asset, tau, lambda, seed, n_validators, n_queries, mode `llm|stylised`.
   - Log toàn bộ config, git hash, seed, model revision, dataset hash, prompt hash.
   - Dùng W&B hoặc local MLflow/CSV logs; với ngân sách thấp, CSV + manifest JSON là đủ.
   - Không chạy full grid ngay; chạy theo cấp:
     - smoke test 10 query,
     - pilot 200 query,
     - mini grid 1 asset x 1 model,
     - full grid.

7. **Thống kê và tái lập kết quả**
   - Viết script sinh bảng/figure từ log duy nhất, không chỉnh tay số liệu.
   - Bootstrap paired CI 10,000 resamples.
   - Holm-Bonferroni paired t-test cho baseline comparisons.
   - Xuất `data/*.csv` cho paper và `figures/*.pdf|png`.
   - Acceptance: chạy lại cùng seed cho kết quả lệch trong tolerance đã định, ví dụ dưới 2%.

8. **AWS/Fabric WAN**
   - Chỉ triển khai WAN sau khi LAN pass.
   - Terraform cho 4 peer `c6i.xlarge`, 1 orderer `t3.medium`, S3 snapshot bucket.
   - Đo riêng: endorsement latency, ordering latency, commit latency, chaincode execution time.
   - Gắn budget alarm AWS trước khi chạy.

## Tài Liệu Nên Đọc

- QRE nền tảng: McKelvey & Palfrey, 1995, “Quantal Response Equilibria for Normal Form Games” [IDEAS/RePEc](https://ideas.repec.org/a/eee/gamebe/v10y1995i1p6-38.html).
- LLM như tác nhân kinh tế: Horton, “Large Language Models as Simulated Economic Agents” [NBER](https://www.nber.org/papers/w31122).
- LLM trong repeated games: Akata et al., 2023 [arXiv](https://arxiv.org/abs/2305.16867).
- Simulating human subjects bằng LLM: Aher, Arriaga, Kalai, ICML 2023 [arXiv](https://arxiv.org/abs/2208.10264).
- Generative agents: Park et al., UIST 2023 [Google Research](https://research.google/pubs/generative-agents-interactive-simulacra-of-human-behavior/).
- Prompt injection: Greshake et al., 2023 [paper page](https://awesome-llm-papers.github.io/publications/greshake2023not/); BIPIA benchmark [arXiv](https://arxiv.org/abs/2312.14197).
- Hyperledger Fabric chaincode lifecycle: official docs [Hyperledger Fabric 2.5](https://hyperledger-fabric.readthedocs.io/en/release-2.5/chaincode_lifecycle.html).
- Hyperledger Fabric architecture paper: Androulaki et al., 2018 [arXiv](https://arxiv.org/abs/1801.10228).
- Oracle systems: Chainlink whitepaper/resources [Chainlink](https://chain.link/whitepaper), Pyth low-latency oracle docs [Pyth](https://legacy.pyth.network/).
- Proper scoring rules/proxy scoring: Witkowski et al., AAAI 2017 [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/10590).
- Data docs: EIA API [official](https://www.eia.gov/opendata/documentation.php), FRED Case-Shiller [FRED](https://fred.stlouisfed.org/release?rid=199), yfinance API reference [docs](https://ranaroussi.github.io/yfinance/reference/index.html).

## Assumptions Và Mặc Định

- Làm v1 theo hướng tái lập paper, không mở rộng sang zkML, multi-region WAN, closed-weight models.
- Ưu tiên LAN + simulation trước, AWS sau.
- Dùng open-weight LLM qua vLLM; nếu không có GPU đủ mạnh, giảm grid và dùng stylised sampler sau calibration.
- Mọi số liệu trong paper phải sinh từ script và log, không nhập tay.
- Trước full experiment bắt buộc chạy pilot throughput vì cost estimate trong guide cần được xác minh bằng batching thực tế.
- Clarifications locked during implementation:
  - `news_context` is currently not used as a real external news signal. Each query stores an explicit deterministic note saying that experiments use only the prior 7-day OHLC window. If news is added later, raw dated snippets must be saved, hashed, and included in prompt hashes before LLM calibration.
  - Pilot calibration must be run for all three target LLMs from `experiment_guide.md`: `meta-llama/Meta-Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, and `Qwen/Qwen2.5-7B-Instruct`.
  - The WAN benchmark uses 4 physical Fabric peer instances and 64 logical validator identities, mapped by default as 16 logical validators per peer. Do not report this as 64 physical EC2 validators.
  - Statistical reporting must include Cohen's d with the planned practical-effect target `d >= 0.5`, in addition to paired-bootstrap CI and Holm-Bonferroni tests.
