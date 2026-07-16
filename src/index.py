import json
from pathlib import Path

path_data = Path("/proj/data/19_Chli_Gaesitschachen")
path_out = Path("/proj/output")


def index_directory(
    dir_path: str | Path,
    json_out_path: str | Path
):
  """Get paths of all """

  # parse arguments
  dir_path = Path(dir_path)
  pdf_path_iter = dir_path.rglob("*.pdf")

  # loop through given dirs subdirectories
  pdf_index_records: list[dict] = []
  for i, path in enumerate(pdf_path_iter):
    dict_record = {"id": i, "path": str(path)}
    pdf_index_records.append(dict_record)

  # save to json
  with open(path_out / "index.json", "w", encoding="utf-8") as out:
    json.dump(pdf_index_records, out, ensure_ascii=False, indent=2)

if __name__ == "__main__":
  index_directory(
    dir_path=path_data,
    json_out_path=path_out
  )