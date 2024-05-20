import os
from multiprocessing import cpu_count
import gunicorn_config


workers = 1
threads = int(os.getenv("NUM_THREADS", str(cpu_count() * 1)))


bind = '0.0.0.0:80'
timeout = os.getenv("GUNICORN_TIMEOUT", str(5 * 60))

