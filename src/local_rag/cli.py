"""Command-line interface: one sub-command per feature of the RAG."""

import typer
from langchain_ollama import ChatOllama, OllamaEmbeddings

LLM_MODEL = "llama3.2"
EMBED_MODEL = "bge-m3"

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.callback()
def main() -> None:
    """Ask questions about your PDF files, answered by a local LLM."""


@app.command()
def smoke() -> None:
    """Check that LangChain can reach both Ollama models."""
    llm = ChatOllama(model=LLM_MODEL, validate_model_on_init=True)
    print(f"[{LLM_MODEL}] ", end="", flush=True)
    for chunk in llm.stream("In one sentence: what is Age of Empires II?"):
        print(chunk.text, end="", flush=True)
    print()

    embeddings = OllamaEmbeddings(model=EMBED_MODEL, validate_model_on_init=True)
    vector = embeddings.embed_query("Build houses to raise the population limit.")
    print(f"[{EMBED_MODEL}] a vector of {len(vector)} numbers: {vector[:3]} ...")


if __name__ == "__main__":
    app()
