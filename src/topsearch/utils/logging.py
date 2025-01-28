import logging
import os
from multiprocessing import current_process

def configure_logging():
    logging.basicConfig(format='%(asctime)s %(name)-16s %(levelname)-8s %(message)s',
                        level=os.getenv("LOGLEVEL","INFO").upper(),
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=f"logfile_{current_process().name}")