import logging
import yaml

import torch

from parameter_parser import parameter_parser
from exp.exp_memguard_siamese import ExpClassMemInferSiamese
from exp.exp_memguard_meta import ExpClassMemInferMeta
import config


def config_logger(save_name):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # create file handlers
    fh1 = logging.FileHandler(config.LOG_PATH + save_name + '.txt', 'w')
    fh1.setLevel(logging.INFO)
    fh1.setFormatter(formatter)
    logger.addHandler(fh1)


def main(args):
    # config the logger
    logger_name = "_".join((args['exp'], str(args['cuda'])))
    config_logger(logger_name)
    logging.info(logger_name)

    torch.set_num_threads(args["num_threads"])
    torch.cuda.set_device(args["cuda"])

    # subroutine entry for different methods
    if args['exp'] == 'class_mem_infer_siamese':
        ExpClassMemInferSiamese(args)
    elif args['exp'] == 'class_mem_infer_meta':
        ExpClassMemInferMeta(args)
    else:
        raise Exception('unsupported attack')


if __name__ == "__main__":
    args = parameter_parser()
        
    main(args)
