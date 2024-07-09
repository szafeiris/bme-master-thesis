from pathlib import Path
from typing import Any
from numpy.typing import NDArray
from pydantic import BaseModel


class PicaiPipelineContext(BaseModel):
    dataset: str = None
    isFixedBinWidth: bool = True
    binCount: int = 32
    normallizeScale: int | float = 100
    
    radiomicsFile: Path = None
    radiomics: Any = None