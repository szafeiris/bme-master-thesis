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
        
        if isinstance(obj, np.uint16) or isinstance(obj, np.uint32) or isinstance(obj, np.uint64) or isinstance(obj, np.int32):
            return int(obj)
        
        if isinstance(obj, np.float32):
            return float(obj)
            
        return JSONEncoder.default(self, obj)
    