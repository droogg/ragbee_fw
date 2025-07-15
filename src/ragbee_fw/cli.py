import pickle
from pathlib import Path

import typer

from ragbee_fw import DIContainer, load_config
from ragbee_fw.infrastructure.retriever.bm25_client import BM25Client

app = typer.Typer(help="RagBee CLI")

INDEX_FILE = Path("bm25_index.pkl")


def build_container(config_path: str) -> DIContainer:
    return DIContainer(load_config(config_path))


@app.command("ingest", help="Build and save index from documents in PATH")
def ingest(
    config_path: str = typer.Argument(..., help="Path to app congiguration file"),
):
    container = build_container(config_path)

    _ = container.build_ingestion_service()

    retriever = container._cache.get("retriever_with_index")
    if not retriever:
        typer.secho("Error: Retriever not built correctly", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    with INDEX_FILE.open("wb") as f:
        pickle.dump(retriever, f)

    typer.secho(
        f"✅ The index was successfully built and saved to {INDEX_FILE}",
        fg=typer.colors.GREEN,
    )


@app.command("ask", help="❓Ask a question and get an answer from the index")
def ask(
    config_path: str = typer.Argument(..., help="Path to app congiguration file"),
    query: str = typer.Argument(..., help="Text of your question"),
):
    if not INDEX_FILE.exists():
        typer.secho(
            f"Error: Index not found. Please run `rag ingest PATH` first.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    with INDEX_FILE.open("rb") as f:
        retriever: BM25Client = pickle.load(f)

    container = build_container(config_path)
    config = container.config

    container._cache["retriever_with_index"] = retriever

    service_ans = container.build_answer_service()
    answer = service_ans.generate_answer(query, top_k=config.retriever.top_k)

    typer.secho(answer, fg=typer.colors.CYAN)


@app.command("shell", help="💬 Interactive session: index PATH and accept questions")
def shell(
    config_path: str = typer.Argument(..., help="Path to app congiguration file"),
):
    container = build_container(config_path)
    config = container.config
    service_ans = container.build_app_service()

    typer.secho(
        f"🤖 RAGbee shell is ready. Index from «{config.data_loader.path}» is built.",
        fg=typer.colors.GREEN,
    )
    typer.secho("Enter question (Ctrl-D to exit): ", fg=typer.colors.MAGENTA)

    try:
        while True:
            query = typer.prompt(">> ")
            if not query.strip():
                continue
            ans = service_ans.generate_answer(query, top_k=config.retriever.top_k)
            typer.secho(ans, fg=typer.colors.CYAN)
    except (EOFError, KeyboardInterrupt):
        typer.secho("\n👋 See you later!", fg=typer.colors.YELLOW)


if __name__ == "__main__":
    app(prog_name="rag")
