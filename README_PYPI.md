# RAGBee FW

*Lightweight Retrieval-Augmented-Generation framework for SMEs.*

RAGBee helps you **load documents → split → retrieve → generate**  
with any LLM (OpenAI, HF Inference API, local vLLM/Triton).  
Ports & Adapters + DI give you clean extensibility; CLI lets you run the pipeline in three commands.

---

## ✨ Features

- **Clean architecture** — `core.ports` (contracts) + `infrastructure.*` (plug-ins)  
- **Dependency Injection** — YAML config ⇒ container wires loader / splitter / retriever / LLM  
- **CLI (`ragbee_cli`)** — `ingest`, `ask`, `shell` out of the box  
- **LLM agnostic** — OpenAI, HuggingFace Hub, vLLM, Triton … or your own adapter  
- **Composable** — embed in FastAPI, Telegram-bot, Streamlit, Airflow DAG  
- **MIT license** — free for commercial use

---

## 🚀 Quick start

```bash
pip install ragbee-fw           # 1. install

ragbee_cli ingest config.yml    # 2. build index
ragbee_cli ask config.yml "Что такое RAG?"   # 3. ask
ragbee_cli shell config.yml     # …or interactive REPL
```

#### config.yml (minimal):

```yaml
data_loader:
  type: file_loader
  path: ./data

text_chunker:
  type: recursive_splitter
  chunk_size: 500
  chunk_overlap: 100

retriever:
  type: bm25
  top_k: 3

llm:
  type: hf
  model_name: gpt-3.5-turbo
  token: ${env:HF_TOKEN}
```

## 🧑‍💻 Python API

```python

from ragbee_fw import DIContainer, load_config

cfg = load_config("config.yml")
container = DIContainer(cfg)

# 1) build / update index
ingestion = container.build_ingestion_service()
ingestion.build_index()                # or .update_index()

# 2) answer questions
answer = container.build_answer_service()
print(answer.generate_answer("What is RAG?", top_k=3))
```

## 🗺 Architecture (Clean / Hexagonal)

```
┌───────────────────┐
│      CLI / UI     │  ←  FastAPI, Streamlit, Telegram Bot …
└─────────▲─────────┘
          │ adapter
┌─────────┴─────────┐
│    Application    │  ←  DI container, services
└─────────▲─────────┘
          │ ports
┌─────────┴─────────┐
│      Core         │  ←  pure dataclasses, protocols
└─────────▲─────────┘
          │ adapters
┌─────────┴─────────┐
│ Infrastructure    │  ←  FS-loader, Splitter, BM25, HF LLM …
└───────────────────┘
```

## 📚 Documentation

Docs & API — [README](https://github.com/droogg/ragbee_fw/blob/main/README.md)

Examples — [example/](https://github.com/droogg/ragbee_fw/tree/main/example)


## 🤝 Contributing

1. Fork → clone → poetry install

2. Format code with black . && isort .

3. Submit PR → CI will run lint & tests

See ```CONTRIBUTING.md``` for full guide.

### 📜 License
MIT © V.A. Shevchenko — free for any purpose, commercial or private.
