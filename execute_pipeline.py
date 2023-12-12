from BMEMasterThesis.utils import log, Datasets
from BMEMasterThesis.pipelines import Pipeline, FullCombinationPipeline

from multiprocessing import Process


def executePipeline(pipeline: Pipeline):
    pipeline.run()

def executeFullCombinationPipelinesAsync():
    originalPipeline: Pipeline = FullCombinationPipeline(Datasets.ORIGINAL)
    originalNormalizedPipeline: Pipeline = FullCombinationPipeline(Datasets.ORIGINAL_NORMALIZED)
    n4Pipeline: Pipeline = FullCombinationPipeline(Datasets.N4)
    n4NormalizedPipeline: Pipeline = FullCombinationPipeline(Datasets.N4_NORMALIZED)
    fatNormalizedPipeline: Pipeline = FullCombinationPipeline(Datasets.FAT_NORMALIZED)
    muscleNormalizedPipeline: Pipeline = FullCombinationPipeline(Datasets.MUSCLE_NORMALIZED)
    
    processes = [
        Process(target=executePipeline, args=(originalPipeline,)),
        Process(target=executePipeline, args=(originalNormalizedPipeline,)),
        Process(target=executePipeline, args=(n4Pipeline,)),
        Process(target=executePipeline, args=(n4NormalizedPipeline,)),
        Process(target=executePipeline, args=(fatNormalizedPipeline,)),
        Process(target=executePipeline, args=(muscleNormalizedPipeline,)),
    ]
    
    for p in processes:
        p.start()
        
    for p in processes:
        p.join()

if __name__ == '__main__':   
    try:       
        executeFullCombinationPipelinesAsync()
    except Exception as e:
        log.exception(e)
