import numpy as np
import theano
import logging
import time
from datetime import timedelta
from time import strftime

floatX = theano.config.floatX
device = theano.config.device


def get_experiment_name(params):
    """
    Generate an experiment name from the parameters.
    """
    experiment_name = ",".join([
        "%s=%s" % (k, str(v)) for k, v in params.items()
    ])
    return "".join(i for i in experiment_name if i not in "\/:*?<>|")


class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = record.created - self.start_time
        # using timedelta here for convenient default formatting
        elapsed = timedelta(seconds=elapsed_seconds)

        return "%s - %s - %s - %s" % (
            strftime("%H:%M:%S"),
            record.levelname,
            elapsed,
            record.getMessage()
        )


def create_logger(filepath):
    """
    Create a logger.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_formatter


def random_weights(shape):
    """
    Return weights of a given shape, with values randomly
    distributed around zero.
    """
    drange = np.sqrt(6. / (np.sum(shape)))
    return drange * np.random.uniform(low=-1.0, high=1.0, size=shape)


def create_shared(value, name):
    """
    Create a shared object of a numpy array.
    """
    return theano.shared(value=np.array(value, dtype=floatX), name=name)
