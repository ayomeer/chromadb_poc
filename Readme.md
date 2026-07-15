# chromadb_poc

## Overview

Proof of concept project for using chromadb to make vast amounts of PDFs searchable.

## Ollama

Ollama is used to supply the models greating embeddings. Before running the project, an Ollama (local) server needs to be started in a terminal like this:

```bash
ollama serve
```

When running the project on a new machine for the first time, you'll also need to pull the models used through ollama. E.g.

```bash
ollama pull nomic-embed-text-v2-moe
```