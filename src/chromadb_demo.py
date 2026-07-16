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
        model_name="nomic-embed-text-v2-moe",
        timeout=300
    )
    ollama_ef = cast(EmbeddingFunction, ollama_ef) # just so pylance doesn't freak out

    collection = chroma_client.get_or_create_collection(
        name="demo_collection",
        embedding_function=ollama_ef,
    )


    # load collection from json and create chroma collection from it
    # data_path="/proj/output/chroma_data.json"
    # with open(data_path, "r") as file:
    #     collection_data: list[dict] = json.load(file)

    # # flatten pdf doc data elements into lists of same lenth as chunks 
    # ids = []
    # documents = []
    # metadatas = []
    # i: int = 0
    # for element in collection_data:
    #     documents.extend(element["chunks"])
    #     for _ in element["chunks"]:
    #         i += 1
    #         ids.append(str(i))
    #         metadatas.append(element["metadata"])

    # print("adding data to collection.")
    # collection.add(
    #     ids=ids,
    #     documents=documents,
    #     metadatas=metadatas
    # )

    # query chromadb
    n_results=5
    query_texts=[
        "Gefahr für Amphibien beim überqueren von Strassen"
    ]
    results = collection.query(
        query_texts=query_texts,
        n_results=n_results
    )
    
    for i, query_text in enumerate(query_texts):
        print(f"\nQuery results for query {i}: '{query_text}':")
        for j in range(n_results):
            print("----------------------------------------------------")
            print(f"RESULT {j+1}:")
            print("----------------------------------------------------")
            print("metadatas\n", results["metadatas"][i][j])
            print("doucuments:\n", results["documents"][i][j])
            print("istances:\n", results["distances"][i][j])

    # print("Query 2 Results:")
    # print(results["metadatas"][1])
    # print(results["metadatas"][1])
    # print(results["distances"][1])