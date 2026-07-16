from __future__ import annotations

from pathlib import Path
from typing import cast

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
texts: list[str] = cast(list[str], text_splitter.split_text(text))

with open(path_out / "output.txt", "w", encoding="utf-8") as out:
    for text_chunk in texts:
        out.write(text_chunk)

dummy: int = 1
