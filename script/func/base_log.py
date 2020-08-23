import os
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG

def create_logger(exp_version):
    log_dir = os.environ["LOG"]
    save_log_dir = log_dir+f"/{exp_version[:-1]}"
    if not os.path.exists(save_log_dir):
        os.mkdir(save_log_dir)

    log_file = (f"{save_log_dir}/{exp_version}.log")

    # logger
    logger_ = getLogger(exp_version)
    logger_.setLevel(DEBUG)

    # formatter
    fmr = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")

    # file handler
    fh = FileHandler(log_file, "w")
    fh.setLevel(DEBUG)
    fh.setFormatter(fmr)

    # stream handler
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(fmr)

    logger_.addHandler(fh)
    logger_.addHandler(ch)

def get_logger(exp_version):
    return getLogger(exp_version)