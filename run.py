from src.main import app
import os
import logging

here = os.path.dirname(__file__)
LOGGER = logging.getLogger()

log_file_path = os.path.join(here, "app.log")
file_handler = logging.FileHandler(log_file_path)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)

LOGGER.info("Pipeline is initialized!")

file_handler = logging.FileHandler(log_file_path)


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')

