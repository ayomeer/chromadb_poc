# Make sure to run ollama and have the models used pulled to your machine before running!
import json
from pathlib import Path
from typing import cast
import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction


def add_data_to_collection(
    collection: chromadb.Collection, 
    data_path: str | Path
) -> None:
    #parse args
    data_path = Path(data_path)

    

    # load collection from json and create chroma collection from it
    with open(data_path, "r") as file:
        collection_data: list[dict] = json.load(file)

    # flatten pdf doc data elements into lists of same lenth as chunks 
    ids = []
    documents = []
    metadatas = []
    i: int = 0
    for element in collection_data:
        documents.extend(element["chunks"])
        for _ in element["chunks"]:
            i += 1
            ids.append(str(i))
            metadatas.append(element["metadata"])

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    return

if __name__ == "__main__":

    # Set up chromadb objects
    chroma_client = chromadb.PersistentClient(path="./chroma")

    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434",
        model_name="nomic-embed-text-v2-moe"
    )
    ollama_ef = cast(EmbeddingFunction, ollama_ef) # just so pylance doesn't freak out

    collection = chroma_client.get_or_create_collection(
        name="demo_collection",
        embedding_function=ollama_ef,
    )

    # add_data_to_collection(
    #     collection=collection,
    #     data_path="/proj/output/chroma_data.json"
    # )

    # load collection from json and create chroma collection from it
    data_path="/proj/output/chroma_data.json"
    with open(data_path, "r") as file:
        collection_data: list[dict] = json.load(file)

    # flatten pdf doc data elements into lists of same lenth as chunks 
    ids = []
    documents = []
    metadatas = []
    i: int = 0
    for element in collection_data:
        documents.extend(element["chunks"])
        for _ in element["chunks"]:
            i += 1
            ids.append(str(i))
            metadatas.append(element["metadata"])

    print("adding data to collection.")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    # query chromadb
    print("query chromadb")
    results = collection.query(
        query_texts=[
            "Amphibienlaichgebiet Gäsi",
            "Einsprachen"
        ],
        n_results=2
    )
    print(results["documents"])