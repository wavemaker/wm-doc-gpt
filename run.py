from src.main import app
import os
import logging
from logging.config import dictConfig

here = os.path.dirname(__file__)
log_file_path = os.path.join(here, "app.log")

logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LOGGER = logging.getLogger(__name__)

LOGGER.info("Logging framework initialized!")



if __name__ == '__main__':
    app.run(debug=False, port=80, host='0.0.0.0')
