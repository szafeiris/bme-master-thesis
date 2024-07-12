from typing import List

class Datasets:
    ORIGINAL: str = "original"
    ORIGINAL_NORMALIZED: str = "original_norm"
    N4: str = "n4"
    N4_NORMALIZED: str = "n4_norm"
    FAT_NORMALIZED: str = "fat"
    MUSCLE_NORMALIZED: str = "muscle"
    FAT_PIECEWISE_NORMALIZED: str = "fat_piecewise"
    MUSCLE_PIECEWISE_NORMALIZED: str = "muscle_piecewise"
    
    ALL_DATASETS: List[str] = [ ORIGINAL, ORIGINAL_NORMALIZED, N4, N4_NORMALIZED, FAT_NORMALIZED, MUSCLE_NORMALIZED, FAT_PIECEWISE_NORMALIZED, MUSCLE_PIECEWISE_NORMALIZED ]
    ORIGINAL_DATASETS: List[str] = [ ORIGINAL, ORIGINAL_NORMALIZED ]
    N4_DATASETS: List[str] = [ N4, N4_NORMALIZED ]
    FAT_DATASETS: List[str] = [ FAT_NORMALIZED, FAT_PIECEWISE_NORMALIZED ]
    MUSCLE_DATASETS: List[str] = [ MUSCLE_NORMALIZED, MUSCLE_PIECEWISE_NORMALIZED ]
    NORMALIZED_DATASETS: List[str] = [ ORIGINAL_NORMALIZED, N4_NORMALIZED, FAT_NORMALIZED, MUSCLE_NORMALIZED ]
    PIECEWISE_DATASETS: List[str] = [ FAT_PIECEWISE_NORMALIZED, MUSCLE_PIECEWISE_NORMALIZED ]
        
    def prettifyDataset(dataset: str):
        return dataset.upper().replace('_', ' ').replace('NORM', '(Normalized)').replace('PIECEWISE', '(Piece-Wise)')
    
    def simplifyDataset(dataset: str):
        return dataset.replace(' ', '_').replace('(Normalized)', 'NORM').replace('(Piece-Wise)', 'PIECEWISE').lower()
    