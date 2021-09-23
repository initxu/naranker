import os
import time
import argparse
from utils.utils import set_reproducible, setup_logger
from utils.config import get_config
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(description='NAR Training')
    parser.add_argument('--config_file',
                        default=None,
                        type=str,
                        help='training configuration')
    parser.add_argument('--data_path',
                        default=None,
                        type=str,
                        help='Path to load data')
    parser.add_argument('--save_dir',
                        default='./output',
                        type=str,
                        help='Path to save output')
    parser.add_argument('--log_path',
                        default='./train.log',
                        type=str,
                        help='Path to save log')

    args = parser.parse_args()

    return args


def build_arg_and_env(run_args):
    args = get_config(run_args.config_file)

    args.config_file = run_args.config_file
    args.data_path = run_args.data_path
    args.log_path = run_args.log_path

    if not os.path.exists(run_args.save_dir):   # 创建总output文件存放各种实验
        os.makedirs(run_args.save_dir)
    import pdb;pdb.set_trace()
    args.save_dir = os.path.join(run_args.save_dir, args.exp_name + '_' + time.strftime('%Y%m%d%H%M%S', time.localtime()))
    import pdb;pdb.set_trace()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)              # 创建当前实验的文件夹
    
    return args


def main():
    run_args = get_args()
    args = build_arg_and_env(run_args)

    logger = setup_logger(save_path=args.log_path)

    set_reproducible(seed=args.seed)
    logger.info('set global rand seed at {}'.format(args.seed))

    logger.info(args)


if __name__ == '__main__':
    main()