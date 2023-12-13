from BMEMasterThesis.utils import log, Datasets
from BMEMasterThesis.pipelines import Pipeline, FullCombinationPipeline

from multiprocessing import Process


def executePipeline(pipeline: Pipeline):
    pipeline.run()

def executeFullCombinationPipelinesAsync():
    processes = [ Process(target=executePipeline, args=(FullCombinationPipeline(dataset),)) for dataset in Datasets._ALL_DATASETS ]
    for p in processes:
        p.start()
        
    for p in processes:
        p.join()

if __name__ == '__main__':   
    try:       
        executeFullCombinationPipelinesAsync()
    except Exception as e:
        log.exception(e)
