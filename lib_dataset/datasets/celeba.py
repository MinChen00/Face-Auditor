import json
import os
import pickle
import re
import zipfile
import gdown
import torch
from imutils import paths
import PIL.Image as Image
from functools import partial
from typing import Any, Callable, List, Optional, Union, Tuple
import torchvision.transforms as transforms
import pandas
import numpy as np

torch.manual_seed(0)
import config


def clean_dataset(rawdata_dir, cleaned_list_name):
    cleaned_list = []
    fp = open(cleaned_list_name, 'r')
    for line in fp.readlines():
        cleaned_list.append(line.split()[0])
    cleaned_list = set(cleaned_list)
    for img in paths.list_images(rawdata_dir):
        img_name = img[-10:]
        if img_name not in cleaned_list:
            print("remove image %s" % img_name)
            os.remove(img)
    print("Finishing Cleaning!")


def dependSecond(elem):
    return elem[1]


def sort_dataset(cleaned_list_name, num_img_list):
    num_imgs, identity_list, clean_img_list, img_ids = [], [], [], []
    fp = open(cleaned_list_name, 'r')
    for line in fp.readlines():
        clean_img_list.append(line.split()[0])
        identity_list.append(line.split()[1])

    id_2_imgs_dict = {}
    for user in set(identity_list):
        id_2_imgs_dict[user] = [clean_img_list[i] for i in np.where(np.array(identity_list) == user)[0]]
    with open(config.RAW_DATA_PATH + 'celeba/id_2_imgs_CelebA.json', 'w') as fp:
        json.dump(id_2_imgs_dict, fp, sort_keys=True, indent=4)

    unique_elements, counts_elements = np.unique(np.array(identity_list), return_counts=True)
    for id, count in zip(unique_elements, counts_elements):
        num_imgs.append((id, count))
    num_imgs.sort(key=dependSecond, reverse=True)
    num_imgs = [str(i) + '\n' for i in num_imgs]
    with open(num_img_list, 'w') as fp:
        fp.writelines(num_imgs)
    print("Finishing Sorting!")


def cut_dataset(topn):
    num_img_list = os.path.join(config.RAW_DATA_PATH, "celeba", 'celeba_num_imgs.txt')
    fp = open(num_img_list, 'r')
    num_imgs = fp.readlines()
    user_list = [int(re.findall('\d+', num_imgs[i])[0]) for i in range(len(num_imgs))]
    return user_list[:topn]


class CelebA(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, is_download=False, topn=6348, num_imges=20, mode='none') -> None:
        self.root = root
        self.transform = transform
        self.mode = mode
        if is_download:
            self.download_dataset()

        fn = partial(os.path.join, self.root)
        self.img_2_id = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        if topn is not None:
            user_list = cut_dataset(topn)
            self.identity = torch.as_tensor(user_list).reshape(-1, 1)
        else:
            self.identity = torch.as_tensor(self.img_2_id.values)

        with open(config.RAW_DATA_PATH + 'celeba/id_2_imgs_CelebA.json', 'r') as fp:
            id_2_imgs = json.load(fp)
        self.filename, self.users = [], {}
        index = 0
        for _, (ids, imgs) in enumerate(id_2_imgs.items()):
            if int(ids) in self.identity.numpy():
                self.filename.extend(imgs)
                self.users[ids] = index
                index += 1

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image_path = os.path.join(self.root, "img_align_celeba", self.filename[index])
        if self.mode in ['low', 'mid', 'high']:
            image_path = os.path.join(self.root, "img_align_celeba" + "_" + self.mode, self.filename[index])

        X = Image.open(image_path)

        # target = self.img_2_id[1][self.filename[index]]
        target = self.users[str(self.img_2_id[1][self.filename[index]])]

        if self.transform is not None:
            X = self.transform(X)

        return X, target

    def __len__(self) -> int:
        return len(self.filename)

    def download_dataset(self):
        base_folder = "celeba/"
        data_root = config.RAW_DATA_PATH + base_folder
        dataset_folder = f'{data_root}/img_align_celeba'
        url = 'https://drive.google.com/uc?id=1cNIac61PSA_LqDFYFUeyaQYekYPc75NH'
        download_path = f'{data_root}/img_align_celeba.zip'

        if not os.path.exists(data_root):
            os.makedirs(data_root)
            os.makedirs(dataset_folder)
        gdown.download(url, download_path, quiet=False)

        with zipfile.ZipFile(download_path, 'r') as ziphandler:
            ziphandler.extractall(dataset_folder)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    os.chdir("../../")
    root = config.RAW_DATA_PATH + "celeba/"
    attr = "identity"
    attr_list = []
    # if isinstance(attr, list):
    #     for a in attr:
    #         if a != "attr":
    #             raise ValueError("Target type \"{}\" is not recognized.".format(a))
    #
    #         num_classes = [8, 4]
    #         # heavyMakeup MouthSlightlyOpen Smiling, Male Young
    #         attr_list = [[18, 21, 31], [20, 39]]
    # else:
    #     if attr == "attr":
    #         num_classes = 8
    #         attr_list = [[18, 21, 31]]
    #     else:
    #         raise ValueError("Target type \"{}\" is not recognized.".format(attr))
    # clean_dataset(root + 'img_align_celeba/', root + 'identity_CelebA.txt')
    # sort_dataset(root + 'identity_CelebA.txt', root + 'celeba_num_imgs.txt')
    dataset = CelebA(root=root, transform=transform)
    print(len(dataset))
