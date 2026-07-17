# Make sure to run ollama and have the models used pulled to your machine before running!
import json
from pathlib import Path
from typing import cast
import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

import umap 
import numpy as np
from matplotlib import pyplot as plt

# -- Functions -------------------------------------------------------------------------
def add_data_to_collection(
    collection: chromadb.Collection, 
    data_path: str | Path
) -> None:
    """Get chunks from json and load into collection (takes some minutes).

    Args:
        collection (chromadb.Collection): Collection to load data into.
        data_path (str | Path): Path to json to load data from.
    """

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

def text_query(
    query_texts: list[str],
    collection: chromadb.Collection,
    n_results: int
) -> None:
    """Query the collection using text strings.

    Args:
        query_texts (list[str]): List of query strings.
        collection (chromadb.Collection): Collection to query.
        n_results (int): Number of results to show.
    """
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

def plot_points(
    points: np.ndarray
):
    

# -- Script Flow Control ---------------------------------------------------------------
LOAD_NEW_DATA = False
EXECUTION_MODE = 'plot' # 'plot' or 'query'

# -- Main ------------------------------------------------------------------------------
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

    if LOAD_NEW_DATA:
        add_data_to_collection(
            collection=collection,
            data_path="/proj/output/chroma_data.json"
        )

    match EXECUTION_MODE:
        case 'query':
            text_query(
                collection=collection,
                n_results=5,
                query_texts=["Gefahr für Amphibien beim überqueren von Strassen"]
            )

        case 'plot':
            # reduce dimensionality w/ UMAP
            reducer = umap.UMAP(
                n_neighbors=15,
                n_components=2,
                metric='cosine'
            )

            results = collection.get(include=['embeddings'])
            points = reducer.fit_transform(results["embeddings"])
            dummy = 1

            # plot 2D space
            plt.scatter(points)
            plt.show()

        case _:
            print("Invalid EXECUTION_MODE.")
            



