from pathlib import Path
from typing import Any
from numpy.typing import NDArray


class PicaiPipelineContext:
    dataset: str = ''
    isFixedBinWidth: bool = True
    binCount: int = 32
    normallizeScale: int | float = 100
    
    radiomicsFile: Path = None
    radiomics: Any = None