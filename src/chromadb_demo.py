# Make sure to run ollama and have the models used pulled to your machine before running! 
# (see Readme)

import json
from pathlib import Path
from typing import cast

import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

import umap
import hdbscan
import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

# -- Constants -------------------------------------------------------------------------

NOISE_LABEL: int = -1

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
            print("Metadata\n", results["metadatas"][i][j], "\n")
            print("Doucument:\n", results["documents"][i][j], "\n")
            print("Distance:\n", results["distances"][i][j], "\n")

def kde_density(points: NDArray):
    # calculate contours using gaussian KDE
    x = points[:, 0]
    y = points[:, 1]
    
    kde = gaussian_kde(np.vstack([x, y]))

    # Create grid
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 200),
        np.linspace(ymin, ymax, 200)
    )

    density = kde(
        np.vstack([xx.ravel(), yy.ravel()])
    ).reshape(xx.shape)

    # plotting 
    plt.contour(xx, yy, density, levels=5)
    plt.show()


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
                n_results=3,
                query_texts=["Gefahr für Amphibien beim überqueren von Strassen"]
            )

        case 'plot':
            # get embeddings 
            collection_data = collection.get(include=['embeddings', 'metadatas'])
            
            # reduce dimensionality w/ UMAP
            reducer = umap.UMAP(
                n_neighbors=15,
                n_components=2,
                metric='euclidean',
                transform_seed=42
            )
            points = reducer.fit_transform(collection_data["embeddings"])
            points = cast(NDArray[np.float64], points) # so pylance doens't freak out

            # cluster low 2D points
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=10,
                min_samples=5,
                cluster_selection_epsilon=0.1,
                metric='euclidean',
                cluster_selection_method='eom' # 'eom' or 'leaf' (more hierarchical)
            )
            clusterer.fit(points)
            labels = clusterer.labels_
            
            # get cluster representatives
            for label in np.unique(labels):
                # skip if label is noise
                if label == NOISE_LABEL:
                    continue 

                label_indeces, = np.where(labels == label)
                label_points = np.take(points, label_indeces, axis=0)
                label_center = np.mean(label_points, axis=0)
                label_representer_idx = np.argmin(np.linalg.norm(label_points-label_center, axis=1))
                
                label_rep_metadata = collection_data["metadatas"][label_representer_idx]
                label_rep_filename = label_rep_metadata["file_path"].split('/')[-1]
                
                print(f"label index: {label}")
                print(f"Representative File: {label_rep_filename}")
                print(f"Full Path: {label_rep_metadata['file_path']}")


            # plotting
            plt.figure(0)
            scatter = plt.scatter(
                points[:, 0], 
                points[:, 1],
                c=labels,
                cmap='tab20'
            )
            plt.colorbar(scatter, label="Cluster ID")
            plt.show()


        case _:
            print("Invalid EXECUTION_MODE.")

    dummy = 1
            



