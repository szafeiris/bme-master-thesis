from pathlib import Path
from typing import Any, List
from numpy.typing import NDArray
from pydantic import BaseModel


class PicaiPipelineContext(BaseModel):
    dataset: str = None
    isFixedBinWidth: bool = True
    binCount: int = 32
    normallizeScale: int | float = 100
    
    radiomicsFile: Path = None
    radiomics: Any = None
    
    X: Any = None
    y: Any = None
    yStratified: Any = None
    trainIndices: Any = None
    testIndices: Any = None
    featureNames: List[str] = None
    patientIds: List[str] = None
    