from pydantic import BaseModel


class GenTreeConfig(BaseModel):
    gossiphs_bin: str = "gossiphs"
    project_path: str = "."
    leaves_limit: int = 10
    leaves_limit_ratio: float = 0.01
    density_ratio: float = 0.9
    infer: bool = False
