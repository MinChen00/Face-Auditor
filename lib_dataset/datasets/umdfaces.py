import zipfile

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import PIL.Image as Image
from imutils import paths
import nums_from_string
import gdown
import os
import torch

import config

torch.manual_seed(0)


def clean_dataset(dir, cleaned_list_name):
    cleaned_list = []
    fp = open(cleaned_list_name, 'r')
    for line in fp.readlines():
        cleaned_list.append(line.split()[0])
    cleaned_list = set(cleaned_list)
    for img in paths.list_images(dir):
        if img[-15:] not in cleaned_list:
            print("remove image %s" % img[-15:])
            os.remove(img)

    print("Finishing Cleaning!")


def dependSecond(elem):
    return elem[1]


def sort_dataset(dir, num_img_list):
    num_imgs = []
    for facedir in os.listdir(dir):
        if os.path.isdir(dir + facedir):
            num_img = len(os.listdir(dir + facedir))
            num_imgs.append((facedir, num_img))

    num_imgs.sort(key=dependSecond, reverse=True)
    num_imgs = [str(i) + '\n' for i in num_imgs]

    with open(num_img_list, 'w') as fp:
        fp.writelines(num_imgs)

    print("Finishing Sorting!")


def cut_dataset(dir, num_img_list, topn):
    fp = open(num_img_list, 'r')
    num_imgs = fp.readlines()
    dir_list = [os.path.join(dir, str(nums_from_string.get_nums(i)[0])) for i in num_imgs]
    # dir_list = [os.path.join(dir, i[2:6]) for i in num_imgs]
    return dir_list[:topn]


class UmdFaces(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, is_download=False, topn=100, num_imges=100):

        self.root = root
        self.transform = transform
        # self.base_folder = "images"
        self.base_folder = ""
        self.img_list = []
        self.label_list = []
        print(os.getcwd())

        if is_download:
            self.download_dataset()

        if topn is not None:
            num_img_list = os.path.join(
                root, self.base_folder, 'umdfaces_num_imgs.txt')
            dir_list = cut_dataset(os.path.join(
                root, self.base_folder), num_img_list, topn)
        else:
            dir_list = os.listdir(os.path.join(root, self.base_folder))

        class_idx = 0
        for dir in dir_list:
            # choose the first 32 images for each identity.
            # for image in os.listdir(os.path.join(root, self.base_folder, dir))[:32]:
            os.chdir(config.WORKDIR)
            for image in os.listdir(dir)[:num_imges]:
                self.img_list.append(os.path.join(dir, image))
                self.label_list.append(class_idx)
            class_idx += 1

        print('load data done')

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = int(self.label_list[index])

        X = Image.open(img_path)

        if self.transform is not None:
            X = self.transform(X)

        return X, label

    def __len__(self):
        return len(self.img_list)

    def download_dataset(self):
        base_folder = ""
        data_root = config.RAW_DATA_PATH + base_folder
        dataset_folder = f'{data_root}/'
        url = 'https://drive.google.com/uc?id=13IDdIMqPCd8h1vWOYBkW6T5bjAxwmxm5'
        download_path = f'{data_root}/umdfaces.zip'

        if not os.path.exists(data_root):
            os.makedirs(data_root)
            os.makedirs(dataset_folder)
        gdown.download(url, download_path, quiet=False)

        with zipfile.ZipFile(download_path, 'r') as ziphandler:
            ziphandler.extractall(dataset_folder)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    os.chdir("../../")
    root = config.WORKDIR + config.RAW_DATA_PATH

    # 1. clean the raw dataset using cleaned_list.txt
    # clean_dataset(root + 'umdfaces/', root + 'umdfaces/images/cleaned_list.txt')
    # 2. sort sub-folders by the number of images in these sub-folders (reverse order)
    sort_dataset(root + 'umdfaces/images/images_low/', root + 'umdfaces/images/images_low/umdfaces_num_imgs.txt')
    # 3. generate dataset (you can choose top n sub-folder to generate your dataset)
    dataset = UmdFaces(root=root + 'umdfaces/images/images_low/', transform=transform, topn=100)
    print(len(dataset))
