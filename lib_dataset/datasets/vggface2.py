import zipfile

import gdown
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import PIL.Image as Image
from imutils import paths
import os
import torch
import config

torch.manual_seed(0)


def clean_dataset(vggface2_dir, cleaned_list_name):
    cleaned_list = []
    fp = open(cleaned_list_name, 'r')
    for line in fp.readlines():
        cleaned_list.append(line.split()[0])
    cleaned_list = set(cleaned_list)
    for img in paths.list_images(vggface2_dir):
        img_name = img[-19:-12]+"\\"+img[-11:]
        if img_name not in cleaned_list:
            print("remove image %s" % img_name)
            os.remove(img)

    print("Finishing Cleaning!")


def dependSecond(elem):
    return elem[1]


def sort_dataset(vggface2_dir, num_img_list):
    num_imgs = []
    for facedir in os.listdir(vggface2_dir):
        if os.path.isdir(vggface2_dir + facedir):
            num_img = len(os.listdir(vggface2_dir + facedir))
            num_imgs.append((facedir, num_img))

    num_imgs.sort(key=dependSecond, reverse=True)
    num_imgs = [str(i) + '\n' for i in num_imgs]

    with open(num_img_list, 'w') as fp:
        fp.writelines(num_imgs)

    print("Finishing Sorting!")


def cut_dataset(vggface2_dir, num_img_list, topn):
    fp = open(num_img_list, 'r')
    num_imgs = fp.readlines()
    dir_list = [os.path.join(vggface2_dir, i[2:9]) for i in num_imgs]
    return dir_list[:topn]


class VGGFace2(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, topn=100, is_download=False, num_imges=32):

        self.root = root
        self.transform = transform
        self.base_folder = "train/"
        self.img_list = []
        self.label_list = []

        if is_download:
            self.download_dataset()

        if topn is not None:
            num_img_list = os.path.join(root, 'vggface_num_imgs.txt')
            dir_list = cut_dataset(os.path.join(root, self.base_folder), num_img_list, topn)
        else:
            dir_list = os.listdir(os.path.join(root, self.base_folder))

        class_idx = 0
        for dir in dir_list:
            # choose the first 32 images for each identity.
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
        ## Fetch data from Google Drive
        data_root = self.root
        url = 'https://drive.google.com/uc?id=1jdZw6ZmB7JRK6RS6QP3YEr2sufJ5ibtO'
        download_path = f'{data_root}Vggface2.zip'
        dataset_folder = data_root + "vggface2"

        if not os.path.exists(data_root):
            os.makedirs(data_root)

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
    root = config.RAW_DATA_PATH

    # 1. clean the raw dataset using cleaned_list.txt
    clean_dataset(root + 'vggface2/train/', root + 'vggface2/VGGFACE2_cleandata_5pts.txt')
    # 2. sort sub-folders by the number of images in these sub-folders (reverse order), save in file
    sort_dataset(root + 'vggface2/train/', root + 'vggface2/vggface_num_imgs.txt')
    # 3. generate dataset (you can choose top n sub-folder to generate your dataset)
    dataset = VGGFace2(root=root + 'vggface2/', transform=transform, topn=5260)
    print(len(dataset))
