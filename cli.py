# import os
import pickle

# import sys
from pathlib import Path

# sys.path.append(os.path.abspath(path=".."))
import typer

from ragbee_fw.core.services.answer_service import AnswerService
from ragbee_fw.core.services.ingestion_service import IngestionService
from ragbee_fw.infrastructure.data_loader.file_loader import FileSystemLoader
from ragbee_fw.infrastructure.llm_clients.huggingface_client import (
    HuggingFaceInferenceAdapter,
)
from ragbee_fw.infrastructure.retriever.bm25_client import BM25Client
from ragbee_fw.infrastructure.text_splitter.recursive_text_splitter import (
    RecursiveTextSplitter,
)

app = typer.Typer(help="RagBee CLI")

INDEX_FILE = Path("bm25_index.pkl")


@app.command("ingest", help="Build and save index from documents in PATH")
def ingest(
    path: str = typer.Argument(
        ..., help="Path to file or directory with documents (txt)"
    ),
):
    loader = FileSystemLoader()
    chunker = RecursiveTextSplitter()
    retriever = BM25Client()

    service = IngestionService(loader=loader, chunker=chunker, retriever=retriever)
    retriever = service.build_index(path)

    with INDEX_FILE.open("wb") as f:
        pickle.dump(retriever, f)

    typer.secho(
        f"✅ The index was successfully built and saved to {INDEX_FILE}",
        fg=typer.colors.GREEN,
    )


@app.command("ask", help="❓Ask a question and get an answer from the index")
def ask(
    query: str = typer.Argument(..., help="Text of your question"),
    top_k: int = typer.Option(
        5, "--top_k", "-k", help="How many fragments to get for context"
    ),
    HF_token: str = typer.Option(
        None, "--token", "-t", help="Your Hugging Face Token with Inference API Support"
    ),
):
    if not INDEX_FILE.exists():
        typer.secho(
            f"Error: Index not found. Please run `rag ingest PATH` first.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    with INDEX_FILE.open("rb") as f:
        retriever: BM25Client = pickle.load(f)

    llm = HuggingFaceInferenceAdapter(
        model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        provider="cerebras",
        token=HF_token,
    )
    service = AnswerService(retriever=retriever, llm=llm)

    answer = service.generate_answer(query, top_k=top_k)
    typer.secho(answer, fg=typer.colors.CYAN)


@app.command("shell", help="💬 Interactive session: index PATH and accept questions")
def shell(
    path: str = typer.Argument(
        ..., help="Path to file or directory with documents (txt)"
    ),
    top_k: int = typer.Option(
        5, "--top_k", "-k", help="How many fragments to get for context"
    ),
    HF_token: str = typer.Option(
        None, "--token", "-t", help="Your Hugging Face Token with Inference API Support"
    ),
):
    loader = FileSystemLoader()
    chunker = RecursiveTextSplitter()
    retriever = BM25Client()
    service_ingest = IngestionService(loader, chunker, retriever)
    retriever = service_ingest.build_index(path)

    llm = HuggingFaceInferenceAdapter(
        model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        provider="cerebras",
        token=HF_token,
    )
    service = AnswerService(retriever=retriever, llm=llm)
    typer.secho(
        f"🤖 RAGbee shell is ready. Index from «{path}» is built.",
        fg=typer.colors.GREEN,
    )
    typer.secho("Enter question (Ctrl-D to exit): ", fg=typer.colors.MAGENTA)

    try:
        while True:
            query = typer.prompt(">> ")
            if not query.strip():
                continue
            ans = service.generate_answer(query, top_k=top_k)
            typer.secho(ans, fg=typer.colors.CYAN)
    except (EOFError, KeyboardInterrupt):
        typer.secho("\n👋 See you later!", fg=typer.colors.YELLOW)


if __name__ == "__main__":
    app(prog_name="rag")
