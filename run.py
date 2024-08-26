from Pooch.main import app
import os
import logging
from logging.config import dictConfig
from Pooch.bot.doc_chat import WMAssistant


LOGGER = logging.getLogger(__name__)
file__ = os.path.dirname(__file__)

log_file_path = os.path.join(file__, "app.log")
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LOGGER.info("Logging framework initialized!")
assistant = WMAssistant()

if __name__ == '__main__':
    app.run(debug=False, port=80, host='0.0.0.0')

