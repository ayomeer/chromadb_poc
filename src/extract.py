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
        separators=["\n\n", "\n", ". ", " "]
    )

    # read json and fill convert to chromadb digestible format
    chroma_data: list[dict] = []
    for json_record in pdf_index:
        id = json_record["id"]
        path = Path(json_record["path"])

        chunks = []

        # -- Create metadata chunk --
        dirnames = list(path.parts)[3:]
        replace_table = str.maketrans("_-.", "   ")

        metadata_string = "\n".join(dirnames).translate(replace_table)

        chunks.append(metadata_string)

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
        content_chunks = text_splitter.split_text(text)
        filename_words = path.stem.translate(replace_table)
        augmented_content_chunks = [filename_words + "\n" + chunk for chunk in content_chunks]

        chunks.extend(augmented_content_chunks)

        # Create dict for exporting data in chromdadb
        chroma_data_element: dict = {
            "id": str(id),
            "metadata": {"file_path": str(path)},
            "chunks": chunks,
        }
        chroma_data.append(chroma_data_element)


    # Save chroma data export to json
    with open(output_path / "chroma_data.json", "w", encoding="utf-8") as out:
        json.dump(chroma_data, out, ensure_ascii=False, indent=2)

    return

# -- Main --------------------------------------------------------------------------------- #

if __name__ == "__main__":
    write_chroma_collection_from_directory(
        index_path="/proj/output/index.json", output_path="/proj/output"
    )
