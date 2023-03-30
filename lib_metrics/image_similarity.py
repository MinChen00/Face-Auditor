import os
import random
import pickle as pkl

import lpips
import torch
import torchvision
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cosine
from torch.nn import CosineSimilarity
from pytorch_msssim import ms_ssim

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import itertools


class ImageSimilarity:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:' + str(args['cuda']) if torch.cuda.is_available() else 'cpu')
        if self.args['image_similarity_name'] == 'lpips':
            self.LOSS_FN_VGG = lpips.LPIPS(net='vgg', version="0.1").to(self.device)

    def pair_similarity(self, a, b):
        if self.args['image_similarity_name'] == 'lpips':
            return self.lpips_similarity(a, b)
        elif self.args['image_similarity_name'] == 'cosine':
            return self.cosine_similarity(a, b)
        elif self.args['image_similarity_name'] == 'mse':
            return self.mse_similarity(a, b)
        elif self.args['image_similarity_name'] == 'msssim':
            return self.msssim_similarity(a, b)
        elif self.args['image_similarity_name'] == 'none':
            return 0
        else:
            print("Unsupported similarity name.")

    def probe_set_similarity(self, probe_dset):
        print("probe_dset shape", probe_dset.dataset.dataset.x.shape)
        data = probe_dset.dataset.dataset.x
        num_classes = np.unique(probe_dset.dataset.dataset.y).shape[0]
        num_shot_per_class = int(probe_dset.dataset.dataset.y.shape[0] / num_classes)
        image_pairs = list(itertools.combinations(range(int(num_shot_per_class)), 2))
        image_similarities = []
        for i in range(num_classes):
            for _, j in enumerate(image_pairs):
                image_a = data[i*num_shot_per_class+j[0]]
                image_b = data[i*num_shot_per_class+j[1]]
                image_similarities.append(self.pair_similarity(image_a, image_b))

        return image_similarities

    def cosine_similarity(self, a, b):
        a = a.view(1, -1).cpu()
        b = b.view(1, -1).cpu()
        return 1 - cosine(a, b)

    def mse_similarity(self, a, b):
        a = a.view(1, -1).cpu()/256
        b = b.view(1, -1).cpu()/256
        return mean_squared_error(a, b)

    def lpips_similarity(self, a, b):
        return 1 - self.LOSS_FN_VGG(a.to(self.device), b.to(self.device)).item()  # similarity

    def msssim_similarity(self, a, b):
        # structural similarity index measure
        img1 = a.reshape([1] + list(a.shape))
        img2 = b.reshape([1] + list(b.shape))
        img1 /= img1.max()
        img2 /= img2.max()
        # https://en.wikipedia.org/wiki/Structural_similarity#Multi-Scale_SSIM
        ssim_value = msssim(img1, img2, normalize=True).item()
        return ssim_value

    def psnr_similarity(self, a, b):
        pass

    def cdf(self, x, plot=True, *args, **kwargs):
        x, y = sorted(x), np.arange(len(x)) / len(x)
        plt.plot(x, y, *args, **kwargs) if plot else (x, y)
