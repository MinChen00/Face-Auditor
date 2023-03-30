import random

import torchvision
from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import config
from lib_metrics.image_similarity import ImageSimilarity


class SiameseTrainDataset(Dataset):
    def __init__(self, dataset, length):
        self.dataset = dataset
        self.num_classes = len(dataset.labels)
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1, _ = self.dataset.dataset[random.choice(self.dataset.labels_to_indices[idx1])]
            image2, _ = self.dataset.dataset[random.choice(self.dataset.labels_to_indices[idx1])]
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1, _ = self.dataset.dataset[random.choice(self.dataset.labels_to_indices[idx1])]
            image2, _ = self.dataset.dataset[random.choice(self.dataset.labels_to_indices[idx2])]

        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


class SiameseTestDataset(Dataset):
    def __init__(self, dataset, times, way):
        self.dataset = dataset
        self.num_classes = len(dataset.labels)
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None
        # generate image pair from same class
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            self.img1, _ = self.dataset.dataset[random.choice(self.dataset.labels_to_indices[self.c1])]
            img2, _ = self.dataset.dataset[random.choice(self.dataset.labels_to_indices[self.c1])]
        # generate image pair from different class
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            img2, _ = self.dataset.dataset[random.choice(self.dataset.labels_to_indices[c2])]

        return self.img1, img2


class SiameseProbe(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.num_classes = len(dataset.labels)
        self.times = args['probe_times']
        self.shot = args['probe_shot']
        self.case = args['image_similarity_level']
        self.img1 = None
        self.c1 = None
        self.selected_idx = None
        self.device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

        self.img_sim = ImageSimilarity(args=args)

    def _batch_visualize(self, save_path):
        image_list = []
        for i in range(self.shot):
            image_list.append(torch.unsqueeze(self.dataset.dataset[self.selected_idx[i - 1]][0], 0))
        image_list = [torch.unsqueeze(self.img1, 0)] + image_list
        batch_tensor = torch.cat(image_list, dim=0)
        # grid_img = torchvision.utils.make_grid(batch_tensor, nrow=self.shot + 1, normalize=True)
        # plt.imshow(grid_img.permute(1, 2, 0))
        # plt.show()
        # torchvision.utils.save_image(grid_img, fp=save_path +'/%s_%s.jpg' % (self.c1, self.similarity), normalize=True)

    def _batch_visualize_pairs(self, save_path):
        image_list = []
        for idx in range(self.shot):
            image_list.append(torch.unsqueeze(self.dataset.dataset[self.selected_idx1[idx]][0], 0))
            image_list.append(torch.unsqueeze(self.dataset.dataset[self.selected_idx2[idx]][0], 0))
        batch_tensor = torch.cat(image_list, dim=0)
        grid_img = torchvision.utils.make_grid(batch_tensor, nrow=2, normalize=True)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
        torchvision.utils.save_image(grid_img, fp=save_path + '/%s_%s.jpg' % (self.c1, self.similarity), normalize=True)

    def _image_level_similarity(self, img1, img2):
        sim = self.img_sim.pair_similarity(img1, img2)
        return sim


class SiameseProbePairs(SiameseProbe):
    def __init__(self, dataset, args):
        super(SiameseProbePairs, self).__init__(dataset, args)

        level = 'top5'
        if args['image_similarity_level'] == 5:
            level = 'last5'

        self.save_path = config.ATTACK_DATA_PATH + "_".join((args['dataset_name'],
                                                             level,
                                                             'distance',
                                                             args['image_similarity_name']))

    def __len__(self):
        return self.times * self.shot

    def __getitem__(self, index):
        idx = index % self.shot

        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            labels_to_indices = self.dataset.labels_to_indices[self.c1]
            pairs = []

            # pairs_iter = min(int(len(labels_to_indices) * len(labels_to_indices) / 2), 1000)
            pairs_iter = 1000
            for i in range(pairs_iter):
                pairs.append(np.random.choice(labels_to_indices, 2))

            # sort pairs distance and select top5 pairs indices
            self.selected_idx1, self.selected_idx2, self.similarity = self._image_pairs_similarity_sort(pairs, labels_to_indices)
            # self._batch_visualize_pairs(self.save_path)

        img1 = self.dataset.dataset[self.selected_idx1[idx]][0]
        img2 = self.dataset.dataset[self.selected_idx2[idx]][0]

        return img1, img2, self.c1, self.similarity[idx]

    def _image_pairs_similarity_sort(self, pairs, labels_to_indices):
        query_img_ids =np.asarray(pairs, dtype=np.int)[:, 0]
        support_img_ids =np.asarray(pairs, dtype=np.int)[:, 1]
        similarity = np.zeros_like(query_img_ids, dtype=np.float)
        for idx, pair in enumerate(pairs):
            img1 = self.dataset.dataset[pair[0]][0]
            img2 = self.dataset.dataset[pair[1]][0]
            dist = self._image_level_similarity(img1, img2)
            similarity[idx] = dist

        # sort_indices = np.argsort(distance)[::-1] # sort distance in a descending way
        sort_indices = np.argsort(similarity)  # sort similarity/distance in an ascending way

        case = self.case
        img_indices, similarities = [], []
        if case == 0:  #  last5 distance, top5 similarity, if sort distance then verse vera.
            img_indices = sort_indices[-self.shot:]
            similarities = similarity[sort_indices[-self.shot:]]  # top5
        elif case == 1:
            img_indices = sort_indices[-55:-50]
            similarities = similarity[sort_indices[-55:-50]]  # top50-55
        elif case == 2:
            img_indices = sort_indices[-105:-100]
            similarities = similarity[sort_indices[-105:-100]]  # top100-105
        elif case == 3:
            img_indices = sort_indices[-155:-150]
            similarities = similarity[sort_indices[-155:-150]]  # top150-155
        elif case == 4:
            img_indices = sort_indices[50:55]
            similarities = similarity[sort_indices[50:55]]  # top250-255
        elif case == 5: # top5 distance, last5 similarity
            img_indices = sort_indices[:5]
            similarities = similarity[sort_indices[0:5]]  # top294-298

        return query_img_ids[img_indices], support_img_ids[img_indices], similarities


class SiameseProbePairs0(SiameseProbe):
    def __init__(self, dataset, args):
        super(SiameseProbePairs0, self).__init__(dataset, args)

    def __len__(self):
        return self.times * self.shot

    def __getitem__(self, index):
        idx = index % self.shot

        # generate image pair from same class
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            labels_to_indices = self.dataset.labels_to_indices[self.c1]
            img1_id = random.choice(labels_to_indices)
            self.img1 = self.dataset.dataset[img1_id][0]

        # return identical image pairs
        return self.img1, self.img1, self.c1, 1.0


class SiameseProbeDatasetBasic(SiameseProbe):
    def __init__(self, dataset, args):
        super(SiameseProbeDatasetBasic, self).__init__(dataset, args)
        self.case = args['image_similarity_level']
        self.dataset = dataset
        self.num_classes = len(dataset.labels)
        self.img1 = None
        self.c1 = None

    def __len__(self):
        return self.times * self.shot

    def __getitem__(self, index):
        idx = index % self.shot

        # generate image pair from same class
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            labels_to_indices = self.dataset.labels_to_indices[self.c1]
            img1_id = random.choice(labels_to_indices)
            self.img1 = self.dataset.dataset[img1_id][0]

        img2, _ = self.dataset.dataset[random.choice(self.dataset.labels_to_indices[self.c1])]

        return self.img1, img2, self.c1


class SiameseProbeDataset(SiameseProbe):
    def __init__(self, dataset, args):
        super(SiameseProbeDataset, self).__init__(dataset, args)
        self.save_path = config.ATTACK_DATA_PATH + "_".join((args['dataset_name'],
                                                             str(args['shot']),
                                                             "basic"))

    def __len__(self):
        return self.times * self.shot

    def __getitem__(self, index):
        idx = index % self.shot

        # generate image pair from same class
        if idx == 0:
            self.c1 = random.randint(0, self.num_classes - 1)
            labels_to_indices = self.dataset.labels_to_indices[self.c1]
            img1_id = random.choice(labels_to_indices)
            self.img1 = self.dataset.dataset[img1_id][0]
            self.selected_idx, self.similarity = self._image_similarity_sort(img1_id, labels_to_indices)
            # self._batch_visualize()

        img2 = self.dataset.dataset[self.selected_idx[idx - 1]][0]
        basic_similarity = self.similarity

        return self.img1, img2, self.c1, basic_similarity

    def _image_similarity_sort(self, img1_id, labels_to_indices):
        remain_id = np.setdiff1d(labels_to_indices, img1_id)
        img1 = self.dataset.dataset[img1_id][0]
        img1 = img1.to(self.device)
        similarity = np.zeros_like(remain_id, dtype=np.float)

        for idx, id in enumerate(remain_id):
            img = self.dataset.dataset[id][0].to(self.device)
            dist = self._image_level_similarity(img1, img)
            similarity[idx] = dist

        # sort_indices = np.argsort(distance)[::-1] # sort distance in a descending way
        sort_indices = np.argsort(similarity)  # sort distance in a ascending way

        case = self.case
        img_indices, similarities = [], []
        if case == 0:  # top5 distance, last5 similarity
            img_indices = sort_indices[-self.shot:]
            similarities = similarity[sort_indices[-self.shot:]]  # top5
        elif case == 1:
            img_indices = sort_indices[-55:-50]
            similarities = similarity[sort_indices[-55:-50]]  # top50-55
        elif case == 2:
            img_indices = sort_indices[-105:-100]
            similarities = similarity[sort_indices[-105:-100]]  # top100-105
        elif case == 3:
            img_indices = sort_indices[-155:-150]
            similarities = similarity[sort_indices[-155:-150]]  # top150-155
        elif case == 4:
            img_indices = sort_indices[50:55]
            similarities = similarity[sort_indices[50:55]]  # top250-255
        elif case == 5: # top5 similarity, last5 distance
            img_indices = sort_indices[:5]
            similarities = similarity[sort_indices[0:5]]  # top294-298

        return remain_id[img_indices], similarities