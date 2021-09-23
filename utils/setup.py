import os
import sys
import torch
import random
import logging
import numpy as np


def setup_seed(seed=20211117):
    # there are still other seed to set, NASBenchDataset
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # for cpu
    torch.cuda.manual_seed(seed)  # for single cuda
    torch.cuda.manual_seed_all(seed)  # for mult gpus
    torch.backends.cudnn.benchmark = False  # Disable conv opts benchmarking
    torch.backends.cudnn.deterministic = True  # make sure the conv algos are deterministic themselves

def setup_logger(save_path=None, mode='a') -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 所有位于DEBUG级别以上的log信息都会被打印
    logger.propagate = False  # 防止日志记录向上层logger传递
    formatter = logging.Formatter("[%(asctime)s]: %(message)s",
                                  datefmt="%m/%d %H:%M:%S")

    # 将log写入指定的文件的handler
    if save_path is not None:
        if os.path.exists(save_path):
            os.remove(save_path)
        log_file = open(save_path, 'w')
        log_file.close()

        file_handler = logging.FileHandler(save_path, mode=mode)  # append mode
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
