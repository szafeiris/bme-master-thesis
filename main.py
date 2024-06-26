from bme_thesis.pipeline.RadiomicsAnalysisPicaiPipeline import RadiomicsAnalysisPicaiPipeline
from bme_thesis.utils.settings import bmeThesisSettings
from bme_thesis.utils.paths import Paths
from bme_thesis.utils.dataset import Datasets
from bme_thesis.utils.notification import sendNotification
from bme_thesis.logger import getLogger
from bme_thesis.utils.time import TimeUtil

log = getLogger(__name__)

pipeline = RadiomicsAnalysisPicaiPipeline(Datasets.ORIGINAL)
pipeline.run()
