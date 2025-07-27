# RAGBee FW

Lightweight Retrieval-Augmented Generation framework for small and medium projects. RAGBee lets you ingest local documents, build a search index and query an LLM with relevant context through a simple CLI or Python API.

## Features

- **Ports & Adapters architecture.** Core contracts live in `core.ports`; plug-in implementations reside in `infrastructure.*`.
- **Dependency Injection container.** `DIContainer` wires components from a YAML configuration.
- **CLI utility.** `ragbee_cli` provides `ingest`, `ask` and `shell` commands.
- **LLM agnostic.** Works with Hugging Face Inference API by default; custom adapters can be registered.
- **Extensible.** Add your own loaders, splitters or retrievers via the adapter registry.
- **MIT licensed.** Free for commercial use.

## Installation

```bash
pip install ragbee-fw
```

Requires Python 3.10 or newer.

## Quick start

1. Prepare a configuration file (see [example/exampl_app_config.yml](example/exampl_app_config.yml)).
2. Build the index:

```bash
ragbee_cli ingest path/to/config.yml
```

3. Ask a question:

```bash
ragbee_cli ask path/to/config.yml "What is RAG?"
```

Interactive session:

```bash
ragbee_cli shell path/to/config.yml
```

### Minimal configuration

```yaml
data_loader:
  type: file_loader
  path: /workspace/documents/

text_chunker:
  type: recursive_splitter
  chunk_size: 1000
  chunk_overlap: 200
  separators:

retriever:
  type: bm25
  top_k: 5

llm:
  type: HF
  model_name: meta-llama/Llama-4-Scout-17B-16E-Instruct
  token: null
  provider: cerebras
  base_url:
  prompt: |
    You are an intelligent assistant.
    Answer the question based on additional information found
    on the topic of the question in the user knowledge base.
  max_new_tokens: 2048
  return_full_response: True
```

## Python API

```python
from ragbee_fw import DIContainer, load_config

cfg = load_config("config.yml")
container = DIContainer(cfg)

# build / update index
ingestion = container.build_ingestion_service()
ingestion.build_index(cfg.data_loader.path)

# ask questions
answer = container.build_answer_service()
print(answer.generate_answer("What is RAG?", top_k=3))
```

## Architecture

```text
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
│ Infrastructure    │  ←  FS loader, splitter, BM25, HF LLM …
└───────────────────┘
```

The CLI and API rely on the dependency injection container to build services. Adapters are resolved from the registry defined in `DIContainer`.

## 📚 Documentation

Docs & API — [README](https://github.com/droogg/ragbee_fw/blob/main/README.md)

Examples — [example/](https://github.com/droogg/ragbee_fw/tree/main/example)

## 🤝 Contributing

We welcome contributions to RAGBee FW! Here's how to get started:

#### 🔧 1. Set up the project

Clone the repository and install dependencies using Poetry:

```bash
git clone https://github.com/droogg/ragbee_fw.git
cd ragbee_fw
poetry install
```
💡 The project uses PEP 621 and supports Python 3.10+.

#### 🐳 2. (Optional) Use DevContainer

This repo includes a .devcontainer/ setup for VS Code, based on an NVIDIA CUDA image.
You can launch it via the Dev Containers extension to get a ready-to-use environment.

If you plan to use only cloud APIs (like OpenAI or HuggingFace), you can simplify the Dockerfile to remove GPU requirements.

#### 🧼 3. Code style

Before submitting your PR:

Format code:

```bash
black .
isort .
Run tests (if added):
```
```bash
pytest
```

#### 🚀 4. Submit Pull Request

Once you're ready:

1. Push your changes to a fork

2. Open a pull request (PR) against main

#### 📘 Full guide
See [CONTRIBUTING.md](CONTRIBUTING.md) for details on architecture, development tips, and extensibility.

## License

RAGBee FW is released under the MIT license.