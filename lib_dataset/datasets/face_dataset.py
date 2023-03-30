import os
import zipfile
import gdown
import torch
from imutils import paths
import PIL.Image as Image
from functools import partial
from typing import Any, Callable, List, Optional, Union, Tuple
import torchvision.transforms as transforms
import pandas

import config

dataset_GOOGLE_DRIVE_FILE_ID = {
    'celeba': '1cNIac61PSA_LqDFYFUeyaQYekYPc75NH',  # img_align_celeba.zip 1.4G
    'celeba_sp': '1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z',  # img_align_celeba.zip 1.4G
    'celeba_hq': '1badu11NqxGf6qM3PTTooQDJvQbejgbTv',  # img_align_celeba.zip 1.4G
    'utkface': '1T5KH-DWXu048im0xBuRK0WEi820T28B-',  # UTKFace.zip 116M
    'lfw': '1WO5Meh_yAau00Gm2Rz2Pc0SRldLQYigT',  # LFW_align_112 1.1G
    'calfw': '1kpmcDeDmPqUcI5uX0MCBzpP_8oQVojzW',  # calfw_align_112 1.1G *****
    'cplfw': '14vPvDngGzsc94pQ4nRNfuBTxdv7YVn2Q',  # cplfw_align_112 1.1G *****
    'cplfw_raw': '1WipxZ1QXs_Fi6Y5qEFDayEgos3rHDRnS',  # CPLFW.zip 258M
    'vggface': '1jdZw6ZmB7JRK6RS6QP3YEr2sufJ5ibtO',  # VGGFace2 21G *****
    'webface': '1wJC2aPA4AC0rI-tAL2BFs2M8vfcpX-w6',  # casia-maxpy-clean.zip 3.94G *****
    'umdfaces': '13IDdIMqPCd8h1vWOYBkW6T5bjAxwmxm5',  # Umdfaces.zip 3.94G *****
    'ms1m': '1X202mvYe5tiXFhOx82z4rPiPogXD435i'  # ms1m_align_112 26.9G
}

transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def download_dataset(dataset_name, root):
    google_drive_url_prefix = 'https://drive.google.com/uc?id='
    url = google_drive_url_prefix + dataset_GOOGLE_DRIVE_FILE_ID[dataset_name]
    if not os.path.exists(root):
        os.makedirs(root)
    os.chdir(root)
    file_name = gdown.download(url, quiet=True)

    file_path = root + file_name
    with zipfile.ZipFile(file_path, 'r') as ziphandler:
        ziphandler.extractall(file_path.split('.')[0])


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, root, dataset_name, transform=None, download=True):
        self.root = root
        self.transform = transform
        self.dataset_name = dataset_name
        if download:
            download_dataset(self.dataset_name, self.root)
        self.dataset_preprocess()

    def dataset_preprocess(self):
        if dataset_name == 'celeba':
            attr = "attr"
            if isinstance(attr, list):
                for a in attr:
                    if a != "attr":
                        raise ValueError("Target type \"{}\" is not recognized.".format(a))

                    num_classes = [8, 4]
                    # heavyMakeup MouthSlightlyOpen Smiling, Male Young
                    attr_list = [[18, 21, 31], [20, 39]]
            else:
                if attr == "attr":
                    num_classes = 8
                    attr_list = [[18, 21, 31]]
                else:
                    raise ValueError("Target type \"{}\" is not recognized.".format(attr))
            self.attr_list = attr_list
            fn = partial(os.path.join, self.root)
            splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
            identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
            bbox = pandas.read_csv(fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0)
            landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1)
            attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

            mask = slice(None)

            self.filename = splits[mask].index.values
            self.identity = torch.as_tensor(identity[mask].values)
            self.bbox = torch.as_tensor(bbox[mask].values)
            self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
            self.attr = torch.as_tensor(attr[mask].values)
            self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
            self.attr_names = list(attr.columns)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = Image.open(os.path.join(self.root, "img_align_celeba", self.filename[index]))

        target: Any = []
        target_type = ['identity']
        for t, nums in zip(target_type, self.attr_list):
            if t == "attr":
                final_attr = 0
                for i in range(len(nums)):
                    final_attr += 2 ** i * self.attr[index][nums[i]]
                target.append(final_attr)
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


if __name__ == "__main__":
    os.chdir("../../")
    dataset_name = 'celeba'
    root = os.getcwd() + '/' + config.RAW_DATA_PATH + dataset_name + '/'
    dataset = FaceDataset(root=root, dataset_name=dataset_name, download=False)
    print(len(dataset))
