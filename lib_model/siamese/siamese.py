import logging

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from lib_dataset.data_store import DataStore
from lib_dataset.meta_dataset import FilteredMetaDataset
from lib_dataset.siamese_dataset import SiameseTestDataset, SiameseTrainDataset
from opacus import PrivacyEngine
from opacus.utils import module_modification
from opacus.dp_model_inspector import DPModelInspector

from lib_model.siamese.model import SiameseModel
from lib_model.few_shot_base import FewShotBase
from parameter_parser import parameter_parser


class Siamese(FewShotBase):
    def __init__(self, args):
        super(Siamese, self).__init__(args)
        self.logger = logging.getLogger('siamese')
        siamese_model = {
            "full_omniglot": SiameseModel(net_name='net0'),
            "miniimagenet": SiameseModel(net_name='net1'),
            "fc100": SiameseModel(net_name='net1'),
            "webface": SiameseModel(net_name='net1'),
            "umdfaces": SiameseModel(net_name='net1'),
            "celeba": SiameseModel(net_name='net1'),
            "vggface2": SiameseModel(net_name='net1')
        }
        if self.args['image_size'] == 96:
            siamese_model = {
                "full_omniglot": SiameseModel(net_name='net0'),
                "miniimagenet": SiameseModel(net_name='net2'),
                "fc100": SiameseModel(net_name='net2'),
                "webface": SiameseModel(net_name='net2'),
                "umdfaces": SiameseModel(net_name='net2'),
                "celeba": SiameseModel(net_name='net2'),
                "vggface2": SiameseModel(net_name='net2')
            }
        elif self.args['image_size'] == 112:
            siamese_model = {
            "full_omniglot": SiameseModel(net_name='net0'),
            "miniimagenet": SiameseModel(net_name='net4'),
            "fc100": SiameseModel(net_name='net4'),
            "webface": SiameseModel(net_name='net4'),
            "umdfaces": SiameseModel(net_name='net4'),
            "celeba": SiameseModel(net_name='net4'),
            "vggface2": SiameseModel(net_name='net4')
            }
        elif self.args['image_size'] == 224:
            siamese_model = {
                "full_omniglot": SiameseModel(net_name='net0'),
                "miniimagenet": SiameseModel(net_name='net3'),
                "fc100": SiameseModel(net_name='net3'),
                "webface": SiameseModel(net_name='net3'),
                "umdfaces": SiameseModel(net_name='net3'),
                "celeba": SiameseModel(net_name='net3'),
                "vggface2": SiameseModel(net_name='net3')
            }
        if self.args['feature_extractor'] == "SCNN":
            self.model = siamese_model[self.args['dataset_name']].to(self.device)
        else:
            self.model = SiameseModel(net_name=self.args['feature_extractor']).to(self.device)

    def train_model(self, train_dset, val_dset, test_dset):
        self.model.train()

        self.train_loader = DataLoader(train_dset, batch_size=32, shuffle=False, num_workers=1)

        loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True) # default setting for siamesenet
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00006)

        if self.args['is_dp_defense']:
            privacy_engine = PrivacyEngine(
                self.model,
                sample_rate=self.args['sample_rate'],
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.args['sigma'],
                max_grad_norm=self.args['max_per_sample_grad_norm'],
                secure_rng=self.args['secure_rng'],
                clip_per_layer=True,
                experimental=True
            )
            privacy_engine.attach(optimizer)

        for batch_id, (img1, img2, label) in enumerate(self.train_loader, 1):
            self.model.train()
            img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)
            img1, img2, label = Variable(img1), Variable(img2), Variable(label)

            optimizer.zero_grad()
            output = self.model.forward(img1, img2)
            loss = loss_fn(output, label)  # for BCEWithLogitsLoss
            loss.backward()
            optimizer.step()

            if batch_id % 100 == 0:
                self.evaluate_model(val_dset, print_message="Val set")
                self.evaluate_model(test_dset, print_message="Test set")

        model_train_precision, model_test_precision = self.evaluate_model(val_dset), self.evaluate_model(test_dset)
        return model_train_precision, model_test_precision

    def evaluate_model(self, test_dset, print_message="Test set"):

        self.model.eval()
        test_loader = DataLoader(test_dset, batch_size=self.args['way'], shuffle=False, num_workers=1)

        right, error = 0, 0
        for _, (test1, test2) in enumerate(test_loader, 1):
            test1, test2 = Variable(test1.to(self.device)), Variable(test2.to(self.device))

            output = self.model.forward(test1, test2).data.cpu().numpy()
            pred = np.argmax(output)
            if pred == 0:
                right += 1
            else:
                error += 1
        model_test_precision = right * 1.0 / (right + error)
        self.logger.info('%s\tcorrect:\t%d\terror:\t%d\tprecision:\t%f' % (print_message, right, error, model_test_precision))

        return model_test_precision

    def probe_model(self, probe_dset):
        self.model.eval()
        probe_loader = DataLoader(probe_dset, batch_size=self.args['shot'], shuffle=False)
        output, label, basic_distance = [], [], []

        for batch_idx, (test1, test2, class_label, distance) in enumerate(probe_loader, 1):
            # self.logger.info("batch_id %d" % batch_idx)
            test1, test2 = test1.to(self.device), test2.to(self.device)
            output.append(self.model.forward(test1, test2).data.reshape((-1)))
            label.append(class_label[0])
            # basic_distance.append(distance[0]) # for SiameseProbeDataset
            basic_distance.append(distance)  # for SiameseProbePairs

        return torch.stack(output).cpu().numpy(), torch.stack(label).numpy(), torch.stack(basic_distance).numpy()

    def generate_embeddings(self, probe_dset):
        self.model.eval()
        self.model.to(self.device)
        probe_loader = DataLoader(probe_dset, batch_size=self.args['shot'], shuffle=False)
        embeddings, class_labels, basic_distance = [], [], []

        for batch_idx, (test1, test2, class_label, distance) in enumerate(probe_loader, 1):
            test1, test2 = test1.to(self.device), test2.to(self.device)
            embedding1, embedding2=self.model.forward_one(test1), self.model.forward_one(test2)
            embeddings.append(embedding1.cpu().detach().numpy())
            class_labels.append(class_label.cpu().detach().numpy())
        return np.stack(embeddings), np.stack(class_label)


if __name__ == '__main__':
    import os
    os.chdir('../../')    
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)
    args = parameter_parser()
    siamesenet = Siamese(args)

    data_store = DataStore(args)
    target_train_dset, shadow_train_dset, \
    target_train_mem_dset, target_train_nonmem_dset, \
    shadow_train_mem_dset, shadow_train_nonmem_dset, \
    target_test_dset, shadow_test_dset = \
    data_store.load_data() 
    length = args['num_iter'] * args['batch_size']

    if args['is_disjoint_train']:
            train_dset = SiameseTrainDataset(target_train_mem_dset, length)
    else:
        train_dset = SiameseTrainDataset(target_train_dset, length)

    val_indices = np.random.choice(np.arange(len(target_train_mem_dset.labels)),
                                    len(target_test_dset.labels))
    val_dset = FilteredMetaDataset(target_train_mem_dset, val_indices)
    val_dset = SiameseTestDataset(val_dset, times=args['test_times'], way=args['way'])
    test_dset = SiameseTestDataset(target_test_dset, times=args['test_times'], way=args['way'])

    target_train_precision, target_test_precision = siamesenet.train_model(train_dset, val_dset, test_dset)