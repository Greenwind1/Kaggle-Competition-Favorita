import logging


def get_logger(log_file):
    logger_ = logging.getLogger('main')
    logger_.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(levelname)s]%(asctime)s:%(name)s:%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger_.addHandler(fh)
    logger_.addHandler(ch)

    return logger_


def kill_logger(logger_):
    for h in logger_.handlers:
        logger_.removeHandler(h)
    logging.shutdown()
    return


if __name__ == '__main__':
    logger = get_logger('./log/test.log')
    logger.info('test')
    kill_logger(logger)
