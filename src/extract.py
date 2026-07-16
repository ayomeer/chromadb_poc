from __future__ import annotations
from typing import cast

import json
from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


# inputs
path_pdf_ocr: Path = Path(
    "/proj/data/19_Chli_Gaesitschachen/20140730_linthwerk_asphaltierung_zufahrt_vrenelibruecke-walenberg.pdf"
)
path_pdf_natural: Path = Path(
    "/proj/data/19_Chli_Gaesitschachen/07_Information/Offerte_WAM_Druck_2026-0099.pdf"
)

# output
path_out: Path = Path("/proj/output")
path_out.mkdir(parents=True, exist_ok=True)

# -- Get Raw Text ---------------------------------------------------------- #

doc: pymupdf.Document = pymupdf.open(path_pdf_natural)
page: pymupdf.Page = next(iter(doc))
text: str = cast(str, page.get_text())

# -- Chunk Text ------------------------------------------------------------ # 

text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks: list[str] = text_splitter.split_text(text)

for i, chunk in enumerate(chunks):
  with open(path_out / f"output_{i}.txt", "w", encoding="utf-8") as out:
    out.write(chunk)  

# 
  # out.writelines(chunks)

dummy: int = 1
