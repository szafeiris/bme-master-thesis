import json
from pathlib import Path
from typing import Any

from bme_thesis.utils.CustomJSONEncoder import CustomJSONEncoder
from bme_thesis.utils.paths import Paths

class JSON:
    def save(obj: Any, path: str | Path, **kwargs):
        path = Paths.toPath(path)
        json.dump(obj, path.open('w'), cls=CustomJSONEncoder, **kwargs)
    
    def load(path: str | Path, **kwargs) -> Any :
        path = Paths.toPath(path)
        return json.load(path.open('r'), **kwargs)