from __future__ import annotations
from typing import cast

import json
from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -- Script Config ------------------------------------------------------------------------------ #

# Test PDFs
path_pdf_ocr = Path(
    "/proj/data/19_Chli_Gaesitschachen/20140730_linthwerk_asphaltierung_zufahrt_vrenelibruecke-walenberg.pdf"
)
path_pdf_natural = Path(
    "/proj/data/19_Chli_Gaesitschachen/07_Information/Offerte_WAM_Druck_2026-0099.pdf"
)
path_in = path_pdf_natural


# -- Main --------------------------------------------------------------------------------- #


def write_chroma_collection_from_directory(
    index_path: str | Path, output_path: str | Path
) -> None:

    # Parse arguments
    index_path = Path(index_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read Index
    with open(index_path, "r") as file:
        pdf_index: list[dict] = json.load(file)

    # Prepare stuff
    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chroma_data: list[dict] = []
    for record in pdf_index:
        id = record["id"]
        path = record["path"]

        # -- Create content chunks --
        # Read raw text
        try:
            doc: pymupdf.Document = pymupdf.open(path)
            page: pymupdf.Page = next(iter(doc))
            text: str = cast(str, page.get_text())
        except FileNotFoundError:
            print(f"Unable to read file at path {path}. Make sure index is up to date.")
            return

        # Chunk Text

        chunks = text_splitter.split_text(text)

        # Create dict for exporting data in chromdadb
        chroma_data_element: dict = {
            "id": str(id),
            "metadata": {"file_path": str(path)},
            "chunks": chunks,
        }

        # TODO: create metadata chunk

        # add element to list
        chroma_data.append(chroma_data_element)

    # Save chroma data export to json
    with open(output_path / "chroma_data.json", "w", encoding="utf-8") as out:
        json.dump(chroma_data, out, ensure_ascii=False, indent=2)

    return


if __name__ == "__main__":
    write_chroma_collection_from_directory(
        index_path="/proj/output/index.json", output_path="/proj/output"
    )
