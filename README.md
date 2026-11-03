# AegisQA Enterprise v3.0 — AI Quality, OWASP Guardrail & Connected IoT Automation Platform

**A Tier-1 Production-Grade Quality Engineering Platform for LLM Assistants, RAG Pipelines, and BLE/Wi-Fi Connected IoT Ecosystems.**

AegisQA Enterprise v3.0 is built from first principles with **real mathematical retrieval & evaluation algorithms**, a **binary BLE GATT framing protocol stack with CRC-16/CCITT verification**, **Population Stability Index (PSI) statistical drift monitoring**, and a **multi-threaded TCP HTTP OpenAPI 3.1 server**. It runs 100% deterministically out-of-the-box with Python 3.11+ and optionally connects to live OpenAI/Gemini judges and Appium servers.

---

## 1. System Architecture & Engineering Highlights

| Subsystem | Module Path | Production Algorithms & Protocols Implemented |
| :--- | :--- | :--- |
| **Hybrid Information Retrieval** | `aegisqa/retrieval/hybrid_retriever.py` | **Okapi BM25 ($k_1=1.5, b=0.75$)** with Robertson-Sparck Jones IDF + **Sublinear TF-IDF Vector Cosine Similarity** over chunked product manuals. |
| **Mathematical RAG & Agent Evaluators** | `aegisqa/evaluators/metrics_engine.py` | Atomic Claim Decomposition & NLI Entailment (**Faithfulness / Groundedness**), **Average Precision @ K (AP@K)** for Contextual Precision, **Answer Relevancy**, and optional **Live LLM-as-a-Judge** (`gpt-4o-mini` / Gemini). |
| **OWASP LLM Top 10 Security Scanner** | `aegisqa/security/owasp_llm_scanner.py` | Defends against **LLM01** (Direct & Indirect RAG Prompt Injection), **LLM02** (Insecure Output XSS/Command Injection), **LLM06** (Aadhaar, PAN, +91 Phone, Email, AWS Key PII Scrubbing), **LLM07/LLM08** (6-digit PIN authorization gate on smart locks). |
| **Binary BLE GATT & Appium Automation** | `aegisqa/iot_automation/ble_gatt_protocol.py` | Encodes/decodes binary GATT packets (`[0xA5][VER][SEQ][OP][LEN][PAYLOAD][CRC16]`) with genuine **CRC-16/CCITT-FALSE (`0x1021`)** checksums, bit-flip self-healing, **BLE 4.2 (23B) vs BLE 5.0 (247B) MTU fragmentation**, and **2.4GHz vs 5GHz (5180 MHz) RF band validation**. |
| **Statistical Drift & SQL/Athena Warehouse** | `aegisqa/pipeline/statistical_drift.py` | Seeds 500 telemetry traces (`v2.3.0-prod` vs `v2.4.0-rc1`) and computes **Population Stability Index ($	ext{PSI} = \sum (P_i - Q_i)\ln(P_i/Q_i)$)**, **KL Divergence**, **Welch's $t$-statistic**, and **auto-synthesizes production failures into regression tests**. |
| **Live TCP HTTP Gateway & Load Runner** | `aegisqa/server/http_gateway.py` | Spins up a real `ThreadingHTTPServer` with **HMAC-SHA256 JWT authentication**, OpenAPI 3.1 contract validation, and executes concurrent **real-socket TCP load benchmarks** (P50/P95/P99 latency & RPS). |

---

## 2. Quickstart & CLI Usage

```bash
# 1. Run the complete 21-test CI/CD Quality Gate & generate reports/latest_eval_report.json
python3 run_quality_gate.py

# 2. Run the unit & integration test suite directly
python3 -m unittest discover -s tests -v

# 3. Run the CLI tools (statistical drift check, live socket load test, or HTTP server)
python3 -m aegisqa.cli drift
python3 -m aegisqa.cli load-test --requests 60 --concurrency 10
python3 -m aegisqa.cli serve --port 8080
```

---
