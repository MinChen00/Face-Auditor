import logging

import torch


class FewShotBase:
    def __init__(self, args):
        self.logger = logging.getLogger('few_shot_base')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.args = args

    def save_model(self, save_path):
        self.logger.info('saving model')
        temp = self.model.cpu().state_dict()
        torch.save(temp, save_path)

    def load_model(self, save_path):
        self.logger.info('loading model')
        self.logger.info("Current device: %s" % (torch.cuda.current_device()))
        checkpoint = torch.load(save_path, map_location='cpu')
        self.model.load_state_dict(checkpoint)        
