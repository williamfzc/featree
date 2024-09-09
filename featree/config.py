from pydantic import BaseModel


class GenTreeConfig(BaseModel):
    gossiphs_bin: str = "gossiphs"
    project_path: str = "."
    leaves_limit: int = 10
    leaves_limit_ratio: float = 0.01
    density_ratio: float = 0.9
    infer: bool = False
    exclude_regex: str = ""
    csv_file: str = "featree-temp.csv"
    symbol_csv_file: str = "featree-symbols-temp.csv"
    include_symbols: bool = False
