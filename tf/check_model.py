import subprocess
import os
import logging
from Akhil.settings import BASE_DIR

logger = logging.getLogger(__name__)
model_url = 'https://storage.googleapis.com/site-storage-akhil/model.ckpt'


def check_model():
    model_exists = os.path.isfile(str(BASE_DIR)+'/colorizer/tf/models/model.ckpt')
    logger.warning(model_exists)
    if not model_exists:
        logger.warning("Model does not exist, downloading...")
        subprocess.check_output(['wget', model_url, '-O', BASE_DIR + '/colorizer/tf/models/model.ckpt'])
    else:
        logger.info("Model exists")
