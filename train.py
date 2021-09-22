import argparse
from utils import setup_logger
def get_args():
    parser = argparse.ArgumentParser(description='NAR Training')
    parser.add_argument('--data-path', default=None, type=str, help='Path to data')
    parser.add_argument('--output-dir', default='./output', type=str, help='Path to save output')
    parser.add_argument('--log-path', default='./train.log', type=str, help='Path to save log')


    
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    logger = setup_logger(save_path=args.log_path)
    logger.info('sddasfdafasfsafasfaf')
    logger.warning('sadfasfdsafasfasfasfasfdsafassafasfsd')
    logger.warning('123')







if __name__ == '__main__':
    main()