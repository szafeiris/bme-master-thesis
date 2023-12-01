
from json import JSONEncoder

from pathlib import Path
import pandas as pd
import numpy as np


class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        if isinstance(obj, Path):
            return str(obj)
        
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        
        return JSONEncoder.default(self, obj)
    